.PHONY: dev run install lint test clean

dev:
	python3 -m uvicorn main:app --reload

run:
	python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

install:
	pip install -r requirements.txt

lint:
	flake8 main.py stellar.py safety tests/redteam study.py fiqh.py hadith.py confidence.py review.py review_store.py tafsir.py semantic_cache.py hadith_research.py --max-line-length=120 --ignore=E501,W503

test:
	pytest -q tests/redteam tests/test_study.py tests/test_semantic_cache.py tests/test_fiqh.py tests/test_hadith.py tests/test_hadith_research.py tests/test_confidence.py tests/test_review_queue.py tests/test_tafsir.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
