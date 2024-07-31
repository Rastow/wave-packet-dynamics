"""Generate the example navigation."""

from pathlib import Path

import mkdocs_gen_files


nav = mkdocs_gen_files.Nav()


for path in sorted(Path("examples").glob("*.toml")):
    examples_path = path.relative_to("examples").with_suffix("")
    doc_path = path.relative_to("examples").with_suffix(".md")
    full_doc_path = Path("examples", doc_path)

    parts = tuple(
        " ".join(word.capitalize() for word in part.split("-")) for part in examples_path.parts
    )

    nav[parts] = doc_path.as_posix()

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)

with mkdocs_gen_files.open("examples/summary.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
