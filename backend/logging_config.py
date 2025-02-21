LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        # "short": {"format": "%(message)s"},
        "default": {
            "format": "%(asctime)s %(levelname)-.1s %(name)s :: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "loggers": {
        "root": {
            # "level": "DEBUG",  # Dynamically set
            "handlers": ["console"],
        },
        "backend": {
            # "level": "DEBUG",  # Dynamically set
            "handlers": ["console"],
            "propagate": False,  # Do not inherit from parent (root) logger
        },
    },
    "root": {
        "handlers": ["muffle"],
        # "handlers": ["console"],
    },
    "handlers": {
        "muffle": {
            "formatter": "default",
            "class": "logging.NullHandler",
        },
        "console": {
            "formatter": "default",
            "class": "logging.StreamHandler",
        },
    },
}
