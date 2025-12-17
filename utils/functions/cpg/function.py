from .ast import AST


class Function:
    def __init__(self, function):
        self.name = function.get("function")
        self.id = function.get("id", "").split(".")[-1]
        self.indentation = 1

        self.ast = AST(function.get("AST", []) or [], self.indentation)
        self.cfg = AST(function.get("CFG", []) or [], self.indentation)
        self.pdg = AST(function.get("PDG", []) or [], self.indentation)

        self.nodes = dict(self.ast.nodes)

        for extra in (self.cfg.nodes, self.pdg.nodes):
            for n_id, n in extra.items():
                if n_id in self.nodes:
                    # Merge edges (keep existing properties from AST version)
                    self.nodes[n_id].edges.update(n.edges)
                else:
                    self.nodes[n_id] = n

    def __str__(self):
        indentation = self.indentation * "\t"
        return f"{indentation}Function Name: {self.name}\n{indentation}Id: {self.id}\n{indentation}AST:{self.ast}"

    def get_nodes(self):
        return self.nodes

    def get_nodes_types(self):
        return {n_id: node.type for n_id, node in self.nodes.items()}
