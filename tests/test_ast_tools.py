import ast

from tools.ast_tools import (
    Wrapper,
    ast_dump,
    code_to_ast,
    code_to_zss_node,
    get_children,
    get_zss_tree,
)


def test_code_to_ast_returns_python_ast_for_valid_code():
    tree = code_to_ast("x = 1")

    assert isinstance(tree, ast.AST)


def test_code_to_ast_returns_none_for_invalid_code():
    assert code_to_ast("for") is None


def test_code_to_ast_dedents_indented_code():
    tree = code_to_ast("    x = 1")

    assert isinstance(tree, ast.AST)


def test_ast_dump_returns_readable_string():
    dumped = ast_dump(code_to_ast("x = 1"))

    assert isinstance(dumped, str)
    assert "Module" in dumped
    assert "Assign" in dumped


def test_code_to_zss_node_wraps_ast_tree():
    node = code_to_zss_node("x = 1")

    assert isinstance(node, Wrapper)
    assert node.label == "Module"
    assert isinstance(get_children(node), list)
    assert len(get_children(node)) > 0


def test_get_zss_tree_uses_cache_for_same_code():
    cache = {}

    first = get_zss_tree("x = 1", cache)
    second = get_zss_tree("x = 1", cache)

    assert first is second
    assert "x = 1" in cache
