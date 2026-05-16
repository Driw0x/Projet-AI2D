import ast
import textwrap

# Outils AST

def code_to_ast(code):
    """
    Transforme un code Python sous forme de texte en AST.
    AST = arbre syntaxique abstrait.
    """
    try:
        return ast.parse(code)

    except Exception:
        try:
            # normaliser indentation
            code_fixed = textwrap.dedent(code)
            return ast.parse(code_fixed)

        except Exception:
            return None


def ast_dump(t):
    """
    Affiche l'AST de manière lisible.
    Utile pour voir la structure du code.
    """
    return ast.dump(t, indent=2)
    
# Wrapping

class Wrapper:
    """
    Wrapper d'un noeud AST pour la distance de Zhang-Shasha.
    """

    def __init__(self, ast_node, path=None):
        self.label = type(ast_node).__name__
        self.children = []

        if path is None:
            self._path = [self.label]
        else:
            self._path = path

        child_index = 0
        for _, value in ast.iter_fields(ast_node):
            if isinstance(value, ast.AST):
                child_path = self._path + [f"{type(value).__name__}[{child_index}]"]
                self.children.append(Wrapper(value, child_path))
                child_index += 1

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        child_path = self._path + [f"{type(item).__name__}[{child_index}]"]
                        self.children.append(Wrapper(item, child_path))
                        child_index += 1

    def get_path(self):
        return self._path

def code_to_zss_node(code):
    """
    Transforme un code Python en arbre compatible avec Zhang-Shasha.
    """
    tree = code_to_ast(code)
    if tree is None:
        return None
    return Wrapper(tree)

def get_children(node):
    """
    Retourne les enfants d'un noeud compatible Zhang-Shasha.
    """
    return node.children

def get_zss_tree(code, zss_cache):
    if code not in zss_cache:
        try:
            zss_cache[code] = code_to_zss_node(code)
        except Exception:
            zss_cache[code] = None
    return zss_cache[code]