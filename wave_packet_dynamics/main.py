"""Main module."""


def hello(name: str = None) -> None:
    """
    Prints a short hello message in the terminal.

    :param str name: user name which is used in the greeting

    :return: only prints a message, doesn't return anything
    :rtype: None
    """
    if name:
        print(f"Hello, {name}")
    else:
        print("Hello, user of wave_packet_dynamics!")
