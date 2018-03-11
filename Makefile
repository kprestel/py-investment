#.RECIPEPREFIX +=
HOST=127.0.0.1
TEST_PATH=./test/


clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +


test-cmd:
	python -m pytest --verbose  --vcr-record-mode=new_episodes --color=yes \
	--pg-extensions=timescaledb --pg-image=timescaledb/timescaledb:latest \
	--pg-name:pytech-test --pg-reuse \
	$(TEST_PATH)


.PHONY: test tests
test: clean-pyc docker-test test-cmd

docker-test:
	docker-compose -f docker-compose-test.yml up -d

