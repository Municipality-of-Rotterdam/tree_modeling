import os
import stat
from importlib.resources import files

from tree_modeling.logger import logger


class Paths:
    """
    Convenience class for executable paths.
    """

    @staticmethod
    def get_adtree() -> str:
        # Locate AdTree directory relative to the package
        logger.info("Custom install of tree_modeling. Will get AdTree path from the repository.")
        adtree_path = files("tree_modeling").joinpath("../../AdTree/bin/AdTree")

        adtree_path_str = str(adtree_path)

        # Check if the executable exists
        if not adtree_path.is_file():
            logger.warning(
                f"WARNING: No file exists at AdTree path: {adtree_path_str}. Please ensure it is included in the package."
            )

        # Ensure the file has execute permissions
        st = os.stat(adtree_path_str)
        if not (st.st_mode & stat.S_IXUSR):
            logger.info(f"Fixing execute permissions for {adtree_path_str}")
            os.chmod(
                adtree_path_str, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )  # Grant execute permission to user, group, and others

        return str(adtree_path)
