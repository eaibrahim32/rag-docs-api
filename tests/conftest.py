"""Root test config.

Deliberately empty of environment setup. Unit tests configure their fakes in
tests/unit/conftest.py; integration tests point at live services in
tests/integration/conftest.py. Setting env here would leak the unit fakes
(EMBEDDING_BACKEND=hash, CACHE_ENABLED=false) into the integration run and
quietly make it test nothing.
"""
