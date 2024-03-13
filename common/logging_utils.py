import logging

import attr


@attr.frozen
class LRLogger:

    name: str
    level: int = logging.INFO

    _logger: logging.Logger = attr.ib(init=False)

    @_logger.default
    def _initialzie_logger(self) -> logging.Logger:
        logging.basicConfig(
            format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        _l = logging.getLogger(self.name)
        _l.setLevel(self.level)
        return _l

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def critical(self, message: str) -> None:
        self._logger.critical(message)
