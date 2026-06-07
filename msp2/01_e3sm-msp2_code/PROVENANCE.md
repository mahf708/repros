# msp2 code base provenance

This is the E3SM source tree used for the msp2 runs. It is built from the
`maint-3.0` branch with two local patches applied on top.

Base commit (origin/maint-3.0):

    f82d08e57d  Merge branch 'tangq/atm/update-r0125-for-v3.NARRM' into maint-3.0 (PR #7744)

Local patches applied on top, in order:

    cfee5a9bd9  edits for msp1   -> 0001-edits-for-msp1.patch
    4e4dbade05  fix queue issue  -> 0002-fix-queue-issue.patch

Both patch files are included in this folder. The patches are already applied
to the source here; they are retained for reproducibility. To rebuild the tree
from scratch:

    git clone git@github.com:E3SM-Project/E3SM.git
    cd E3SM
    git checkout f82d08e57d
    git apply 0001-edits-for-msp1.patch
    git apply 0002-fix-queue-issue.patch
    git submodule update --init --recursive

Files touched by the patches:

    cime_config/machines/cmake_macros/intel_derecho.cmake   (new)
    cime_config/machines/config_batch.xml
    cime_config/machines/config_machines.xml
    components/eam/src/physics/cam/output_aerocom_aie.F90
    components/eam/src/physics/p3/eam/micro_p3_interface.F90
    share/build/buildlib.spio
