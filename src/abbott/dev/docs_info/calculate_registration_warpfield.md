### Purpose
- **Computes image-based registration** transformations for acquisitions in **HCS** OME-Zarr datasets based on [warpfield](https://github.com/danionella/warpfield).
- Takes warpfield registration recipe, if None is provided default registration parameters are applied.
- Can handle cases where there are more than one embryo / organoid in a FOV if each ROI e.g. embryo / organoid is masked by a linked label (e.g. calculated by`scMultiplex Calculate Object Linking`) and corresponding masking_roi_table. Assumes label_id to be the same across cycles, but soesn't require FOVs to have the same shape.
- Typically used as the first task in a workflow, followed by `Apply Registration (warpfield)`.

### Output
- Calculates transformation **warpmap** for **per (ROI)** and stores the results in a registration subfolder of OME-Zarr container.

### Limitations
- Supports only HCS OME-Zarr datasets, leveraging their acquisition metadata and well-based image grouping.
- Requires CUDA-compatible GPU with Cuda > 12.x .
- For **Pelkmans cluster** the following parameters need to be provided manually during Worker Initialisation:
    ```
    #SBATCH -p gpu-cuda12
    export PATH=/usr/local/cuda-12.1/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
    ```
