# Legacy-compatible plotting code

This package contains the influenza portion of the archived `figures.py` and the complete archived `mplot.py`. It exists so historical notebook imports continue to work while renderers move into topic modules. The unrelated CH505/HIV/Wright-Fisher tail remains only in the dated archive.

Use `influenza_figures.render.legacy_style_context` around calls into this
package. It applies one explicit paper, slide, poster, or grant profile to both
historical modules (including their mutable default-property dictionaries) and
restores the previous settings afterward. Root `figures.py` is now the readable
canonical facade and exposes this module explicitly as `figures.legacy`;
ordinary unadapted function lookup is retained for older imports. Root
`mplot.py` remains a compatibility alias.
