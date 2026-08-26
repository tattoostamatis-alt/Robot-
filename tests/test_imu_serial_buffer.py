"""Regression tests for preserving every BNO085 record in serial chunks."""
import ast
from pathlib import Path


SRC = Path(__file__).parents[1] / 'home_robot/nodes/imu_node.py'
tree = ast.parse(SRC.read_text(encoding='utf-8'))
fn = next(n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == '_pop_complete_line')
namespace = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), 'exec'), namespace)
pop_line = namespace['_pop_complete_line']


def test_multiple_serial_records_are_not_discarded():
    buf = bytearray(b'IMU,first\nIMU,second\n')
    assert pop_line(buf) == b'IMU,first\n'
    assert pop_line(buf) == b'IMU,second\n'
    assert buf == b''


def test_partial_record_is_kept_until_newline_arrives():
    buf = bytearray(b'IMU,partial')
    assert pop_line(buf) is None
    assert buf == b'IMU,partial'
    buf.extend(b',done\n')
    assert pop_line(buf) == b'IMU,partial,done\n'
