"""
A module for specifics related to the library
"""
class AllInvalidError(Exception):
    """Raised by `calculate_(all_)metrics`if all observations or all simulations are NaN(Blank) or negative. """              