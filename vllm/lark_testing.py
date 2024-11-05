from lark import Lark
from lark.indenter import Indenter

# Custom Indenter class to handle indentation
class TreeIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8

grammar = r"""
?start: _NL* featuremodel

featuremodel: "features" _NL [_INDENT feature+ _DEDENT]
feature: NAME _NL (_INDENT group+ _DEDENT)?

group: "or" groupspec          -> or_group
| "alternative" groupspec -> alternative_group
| "optional" groupspec    -> optional_group
| "mandatory" groupspec   -> mandatory_group
| cardinality groupspec   -> cardinality_group

groupspec: _NL _INDENT feature+ _DEDENT

cardinality: "[" INT (".." (INT | "*"))? "]"

%import common.INT
%import common.CNAME -> NAME
%import common.WS_INLINE
%declare _INDENT _DEDENT
%ignore WS_INLINE

_NL: /(\r?\n[\t ]*)+/
"""

input_text = r"""
namespace Server

features
  Server {abstract}
    mandatory
      FileSystem
        or // with cardinality: [1..*]
          NTFS
          APFS
          EXT4
      OperatingSystem {abstract}
        alternative
          Windows
          macOS
          Debian
    optional
      Logging	{
      default,
      log_level "warn" // Feature Attribute
    }

constraints
  Windows => NTFS
  macOS => APFS
"""

# Create the parser with Earley for detailed error reporting
parser = Lark(grammar, parser='lalr', postlex=TreeIndenter(), debug=True)

# Parse the input
try:
    tree = parser.parse(input_text)
    print(tree.pretty())
except Exception as e:
    print(e)