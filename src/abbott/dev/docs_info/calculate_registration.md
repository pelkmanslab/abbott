### Purpose
- **Computes image-based registration** transformations for acquisitions in **HCS** OME-Zarr datasets using the elastix library.
- Needs Elastix profiles to configure the registration.
- Calculates registration transforms e.g. per FOV by providing FOV_ROI_table.
- Can handle cases where there are more than one embryo / organoid in a FOV if each ROI e.g. embryo / organoid is masked by a linked label (e.g. calculated by`scMultiplex Calculate Object Linking`) and corresponding masking_roi_table. Assumes label_id to be the same across cycles, but soesn't require FOVs to have the same shape.
- Typically used as the first task in a workflow, followed by `Apply Registration (elastix)`.

### Output
- Calculates transformation parameters for **per (ROI)** and stores the results in a registration subfolder of OME-Zarr container.

### Limitations
- Supports only HCS OME-Zarr datasets, leveraging their acquisition metadata and well-based image grouping.
