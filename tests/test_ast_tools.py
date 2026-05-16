import ast
from tools.ast_tools import *


def test_code_to_ast_valid_code():
    assert isinstance(code_to_ast("x = 1"), ast.AST)


def test_code_to_ast_invalid_code_returns_none():
    assert code_to_ast("for") is None


def test_ast_dump_returns_string():
    dumped = ast_dump(code_to_ast("x = 1"))
    assert isinstance(dumped, str)
    assert "Module" in dumped


def test_zss_wrapping_helpers():
    node = code_to_zss_node("x = 1")
    assert isinstance(node, Wrapper)
    assert node.label == "Module"
    assert isinstance(get_children(node), list)


def test_get_zss_tree_uses_cache():
    cache = {}
    first = get_zss_tree("x = 1", cache)
    second = get_zss_tree("x = 1", cache)
    assert first is second
