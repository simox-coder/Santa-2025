"""
CLI entry point for Santa 2025 solver.
"""

import click


@click.group()
def main():
    """Santa 2025 Christmas Tree Packing Solver"""
    pass


@main.command()
def download():
    """Download competition data from Kaggle."""
    from .download import download_competition_data
    download_competition_data()


@main.command()
@click.argument('notebook_type', type=click.Choice(['metric', 'getting_started', 'all']))
def pull_notebooks(notebook_type):
    """Pull Kaggle notebooks."""
    from .pull_notebooks import pull_metric_notebook, pull_getting_started_notebook
    
    if notebook_type in ('metric', 'all'):
        pull_metric_notebook()
    if notebook_type in ('getting_started', 'all'):
        pull_getting_started_notebook()


@main.command()
def tune():
    """Run ASHA tuning orchestrator."""
    from .orchestrator import main as orchestrator_main
    orchestrator_main()


@main.command()
def solve():
    """Run solver with best config."""
    from .solve import main as solve_main
    solve_main()


@main.command()
@click.option('--submission', required=True, help='Path to submission CSV')
@click.option('--skip-overlaps', is_flag=True, help='Skip overlap checking')
def validate(submission, skip_overlaps):
    """Validate a submission file."""
    from .validate import validate_submission
    
    valid, details = validate_submission(
        submission,
        check_overlaps=not skip_overlaps
    )
    
    if not valid:
        raise click.ClickException("Validation failed")


@main.command()
@click.option('--submission', required=True, help='Path to submission CSV')
@click.option('--quiet', is_flag=True, help='Only print final score')
def score(submission, quiet):
    """Score a submission file."""
    from .score import score_file, quick_score, print_public_score
    
    if quiet:
        s = quick_score(submission)
        print_public_score(s)
    else:
        result = score_file(submission)
        print()
        print_public_score(result['total_score'])


if __name__ == '__main__':
    main()
