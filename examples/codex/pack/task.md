# pallets__flask-5014

Repository: `pallets/flask`

You are working in a checkout of this project. Resolve the issue described
below by editing the source. Do not modify tests.

Note: your shell's current working directory is already the checked-out root of this repository. Do not prefix file paths with the repository name again (for example use `astropy/io/fits/card.py`, not `astropy/astropy/io/fits/card.py`; use `django/db/models/fields/__init__.py`, not `django/django/django/db/models/fields/__init__.py`). Run `ls` first if unsure.

## Issue

Require a non-empty name for Blueprints
Things do not work correctly if a Blueprint is given an empty name (e.g. #4944).

It would be helpful if a `ValueError` was raised when trying to do that.
