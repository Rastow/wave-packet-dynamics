"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files


nav = mkdocs_gen_files.Nav()

for path in sorted(Path("src").rglob("*.py")):
    module_path = path.relative_to("src").with_suffix("")
    doc_path = path.relative_to("src").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":  # noqa: PLR2004
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
        no_members = True
    elif parts[-1] == "__main__" or parts[-1] == "cli":  # noqa: PLR2004
        continue
    else:
        no_members = False

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as f:
        identifier = ".".join(parts)
        f.write(f"::: {identifier}")
        if no_members:
            f.write("\n    options:\n      members: false")

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)

with mkdocs_gen_files.open("reference/summary.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
