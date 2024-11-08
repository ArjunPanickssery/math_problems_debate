def pytest_addoption(parser):
    parser.addoption(
        "--runhf",
        action="store_true",
        default=False,
        help="Run tests with slow HF models",
    )
