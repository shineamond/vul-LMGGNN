import argparse
import gc
import shutil
from argparse import ArgumentParser
import configs
import utils.data as data
import utils.process as process
import utils.functions.cpg_utils as cpg
import torch
import torch.nn.functional as F
from utils.data.datamanager import loads, train_val_test_split
from models.LMGNN import BertGGCN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from test import test
from collections import Counter

PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def select(dataset):
    result = dataset.loc[dataset['project'] == "FFmpeg"]
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]
    print(len(result))
    #result = result.iloc[11001:]
    #print(len(result))
    # result = result.head(200)

    return result

def CPG_generator():
    """
    Generates Code Property Graph (CPG) datasets from raw data.

    :return: None
    """
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)

    # Here, taking the Devign dataset as an example,
    # specific modifications need to be made according to different dataset formats.
    filtered = data.apply_filter(raw, select)
    filtered = data.clean(filtered)
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        cpg_file = process.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)
    # Create CPG with graphs json files
    json_files = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    for (s, slice), json_file in zip(slices, json_files):
        graphs = process.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()

def Embed_generator():
    """
    Generates embeddings from Code Property Graph (CPG) datasets.

    :return: None
    """
    context = configs.Embed()
    dataset_files = data.get_directory_files(PATHS.cpg)

    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        tokens_dataset = data.tokenize(cpg_dataset)
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")

        cpg_dataset["nodes"] = cpg_dataset.apply(lambda row: cpg.parse_to_nodes(row.cpg, context.nodes_dim), axis=1)
        cpg_dataset["input"] = cpg_dataset.apply(lambda row: process.nodes_to_input(row.nodes, row.target, context.nodes_dim,
                                                                            context.edge_type), axis=1)
        data.drop(cpg_dataset, ["nodes"])
        print(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")
        # write(cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}")
        # write(cpg_dataset[["input", "target","func"]], PATHS.input, f"{file_name}_{FILES.input}")
        data.write(cpg_dataset[["input", "target", "func"]], PATHS.input, f"{file_name}_{FILES.input}")

        del cpg_dataset
        gc.collect()


def train(model, device, train_loader, optimizer, epoch):
    """
    Trains the model using the provided data.

    :param model: The model to be trained.
    :param device: The device to perform training on (e.g., 'cpu' or 'cuda').
    :param train_loader: The data loader containing the training data.
    :param optimizer: The optimizer used for training.
    :param epoch: The current epoch number.
    :return: None
    """

    model.train()
    best_acc = 0.0
    for batch_idx, batch in enumerate(train_loader):
        batch.to(device)

        y_pred = model(batch)
        model.zero_grad()
        # print("y_pred data type:", y_pred.dtype)
        # print("batch.y.squeeze() data type:", batch.y.squeeze().dtype)
        batch.y = batch.y.squeeze().long()
        loss = F.nll_loss(torch.log(y_pred + 1e-12), batch.y)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]/t Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(batch),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
            
def train_kd(model_s1, model_s2, device, train_loader, optimizer_s1, optimizer_s2, epoch, alpha):
    model_s1.train()
    model_s2.train()

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        targets = batch.y.squeeze().long()

        # --- Train S1 (teacher = S2, detach) ---
        optimizer_s1.zero_grad()

        y1, z1 = model_s1.forward_with_node_embeddings(batch)
        ce1 = F.nll_loss(torch.log(y1 + 1e-12), targets)

        with torch.no_grad():
            _, z2_t = model_s2.forward_with_node_embeddings(batch)
        lsp1 = compute_lsp_loss(z_student=z1, z_teacher=z2_t.detach(), edge_index=batch.edge_index)

        loss1 = ce1 + alpha * lsp1
        loss1.backward()
        optimizer_s1.step()

        # --- Train S2 (teacher = S1, detach) ---
        optimizer_s2.zero_grad()

        y2, z2 = model_s2.forward_with_node_embeddings(batch)
        ce2 = F.nll_loss(torch.log(y2 + 1e-12), targets)

        with torch.no_grad():
            _, z1_t = model_s1.forward_with_node_embeddings(batch)
        lsp2 = compute_lsp_loss(z_student=z2, z_teacher=z1_t.detach(), edge_index=batch.edge_index)

        loss2 = ce2 + alpha * lsp2
        loss2.backward()
        optimizer_s2.step()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"Loss1: {loss1.item():.6f} (CE={ce1.item():.4f}, LSP={lsp1.item():.4f}) "
                f"Loss2: {loss2.item():.6f} (CE={ce2.item():.4f}, LSP={lsp2.item():.4f})"
            )


def compute_lsp_loss(z_student, z_teacher, edge_index):
    row, col = edge_index
    device = z_student.device
    N = z_student.size(0)

    diff_t = z_teacher[row] - z_teacher[col]
    sim_t = - (diff_t * diff_t).sum(dim=1)

    diff_s = z_student[row] - z_student[col]
    sim_s = - (diff_s * diff_s).sum(dim=1)

    total_kl = torch.tensor(0.0, device=device)
    count = 0

    for i in range(N):
        mask = (row == i)
        if not mask.any():
            continue

        st = sim_t[mask]
        ss = sim_s[mask]

        st_exp = torch.exp(st - st.max())
        pt = st_exp / (st_exp.sum() + 1e-12)

        ss_exp = torch.exp(ss - ss.max())
        ps = ss_exp / (ss_exp.sum() + 1e-12)

        kl_i = torch.sum(pt * (torch.log(pt + 1e-12) - torch.log(ps + 1e-12)))
        total_kl += kl_i
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)

    return total_kl / count

def validate(model, device, test_loader):
    """
    Validates the model using the provided test data.

    :param model: The model to be validated.
    :param device: The device to perform validation on (e.g., 'cpu' or 'cuda').
    :param test_loader: The data loader containing the test data.
    :return: Tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    for batch_idx, batch in enumerate(test_loader):
        batch.to(device)
        with torch.no_grad():
            y_ = model(batch)

        batch.y = batch.y.squeeze().long()
        test_loss += F.nll_loss(torch.log(y_ + 1e-12), batch.y).item()
        pred = pred = y_.argmax(dim=1)

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy().tolist())

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'], yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    print('Val set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

def count_labels(loader):
    ys = []
    for b in loader:
        ys.extend(b.y.view(-1).cpu().numpy().tolist())
    return Counter(ys)

if __name__ == '__main__':
    parser: ArgumentParser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--prepare', help='Prepare task', required=False)
    parser.add_argument('-cpg', '--cpg', action='store_true', help='Specify to perform CPG generation task')
    parser.add_argument('-embed', '--embed', action='store_true', help='Specify to perform Embedding generation task')
    parser.add_argument('-mode', '--mode', default="train", help='Specify the mode (e.g., train, test)')
    parser.add_argument('-path', '--path', default=None, help='Specify the path for the model')

    args = parser.parse_args()

    if args.cpg:
        CPG_generator()
    if args.embed:
        Embed_generator()

    context = configs.Process()
    input_dataset = loads(PATHS.input)
    # split the dataset and pass to DataLoader with batch size
    train_ds, val_ds, test_ds = train_val_test_split(input_dataset, shuffle = context.shuffle)

    train_loader = train_ds.get_loader(context.batch_size, shuffle = True,  drop_last = False)
    val_loader = val_ds.get_loader(context.batch_size, shuffle = False, drop_last = False)
    test_loader = test_ds.get_loader(context.batch_size, shuffle = False, drop_last = False)

    print("Train labels:", count_labels(train_loader))
    print("Val labels:", count_labels(val_loader))
    print("Test labels:", count_labels(test_loader))

    Bertggnn = configs.BertGGNN()
    gated_graph_conv_args = Bertggnn.model["gated_graph_conv_args"]
    conv_args = Bertggnn.model["conv_args"]
    emb_size = Bertggnn.model["emb_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        model_s1 = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)
        model_s2 = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)

        optimizer_s1 = torch.optim.AdamW(
            model_s1.parameters(),
            lr=Bertggnn.learning_rate,
            weight_decay=Bertggnn.weight_decay
        )
        optimizer_s2 = torch.optim.AdamW(
            model_s2.parameters(),
            lr=Bertggnn.learning_rate,
            weight_decay=Bertggnn.weight_decay
        )

        alpha_kd = Bertggnn.loss_lambda

        best_acc = 0.0
        NUM_EPOCHS = context.epochs
        PATH = args.path
        for epoch in range(1, NUM_EPOCHS + 1):
            train_kd(model_s1, model_s2, device, train_loader, optimizer_s1, optimizer_s2, epoch, alpha_kd)
            acc, precision, recall, f1 = validate(model_s1, device, val_loader)
            if best_acc < acc:
                best_acc = acc
                torch.save(model_s1.state_dict(), PATH)
            print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))

    model_test = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)
    model_test.load_state_dict(torch.load(args.path))
    accuracy, precision, recall, f1 = test(model_test, device, test_loader)


