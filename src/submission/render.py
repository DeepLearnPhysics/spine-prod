"""Render scheduler scripts into immutable submission-attempt directories."""

from pathlib import Path


def render_script(
    client, template_name, attempt_dir, chunk_index, num_chunks, **context
):
    """Render one job script and return its executable path.

    Parameters
    ----------
    client : BatchClient
        Scheduler client providing templates and the script suffix.
    template_name : str
        Site template selected from the resource profile.
    attempt_dir : pathlib.Path
        Existing directory owned by this submission attempt.
    chunk_index : int
        Zero-based chunk index used when naming multiple job scripts.
    num_chunks : int
        Number of scheduler jobs planned for this submission.
    **context
        Fully resolved template arguments, including resource overrides.

    Returns
    -------
    pathlib.Path
        Written script, ready to pass to the scheduler client.
    """
    template = client.load_template(template_name)
    content = template.render(**context)
    name = "submit" if num_chunks == 1 else f"submit_{chunk_index:03d}"
    path = Path(attempt_dir) / f"{name}{client.script_suffix}"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path
