.PHONY: dev run install install-dev lock lint test clean

dev:
	python3 -m uvicorn main:app --reload

run:
	python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

install:
	pip install --require-hashes -r requirements-installer.lock
	pip install --require-hashes -r requirements.lock

install-dev: install
	pip install --require-hashes -r requirements-dev.lock

lock:
	pip-compile --generate-hashes --resolver=backtracking --allow-unsafe --output-file requirements-installer.lock requirements-installer.txt
	pip-compile --generate-hashes --resolver=backtracking --allow-unsafe --output-file requirements.lock requirements.txt
	pip-compile --generate-hashes --resolver=backtracking --allow-unsafe --constraint requirements.lock --output-file requirements-dev.lock requirements-dev.txt
	pip-compile --generate-hashes --resolver=backtracking --allow-unsafe --constraint requirements.lock --output-file requirements-perf.lock requirements-perf.txt
	pip-compile --generate-hashes --resolver=backtracking --allow-unsafe --constraint requirements.lock --output-file requirements-alignment.lock requirements-alignment.txt

lint:
	flake8 main.py stellar.py safety tests/redteam study.py fiqh.py hadith.py confidence.py review.py review_store.py tafsir.py semantic_cache.py retrieval scripts/build_index.py --max-line-length=120 --ignore=E501,W503

test:
	pytest -q tests/redteam tests/test_study.py tests/test_semantic_cache.py tests/test_stream_cache.py tests/test_fiqh.py tests/test_hadith.py tests/test_confidence.py tests/test_review_queue.py tests/test_tafsir.py tests/test_retrieval_chunking.py tests/test_retrieval_index.py tests/test_build_index.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
