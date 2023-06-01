import torch

## rewrite
def dense_layer(SparseTensor, shape=None, min_coordinate=None, contract_stride=True, default_value=0.0):
    r"""Convert the :attr:`MinkowskiEngine.SparseTensor` to a torch dense
    tensor.

    Args:
        :attr:`shape` (torch.Size, optional): The size of the output tensor.

        :attr:`min_coordinate` (torch.IntTensor, optional): The min
        coordinates of the output sparse tensor. Must be divisible by the
        current :attr:`tensor_stride`. If 0 is given, it will use the origin for the min coordinate.

        :attr:`contract_stride` (bool, optional): The output coordinates
        will be divided by the tensor stride to make features spatially
        contiguous. True by default.

    Returns:
        :attr:`tensor` (torch.Tensor): the torch tensor with size `[Batch
        Dim, Feature Dim, Spatial Dim..., Spatial Dim]`. The coordinate of
        each feature can be accessed via `min_coordinate + tensor_stride *
        [the coordinate of the dense tensor]`.

        :attr:`min_coordinate` (torch.IntTensor): the D-dimensional vector
        defining the minimum coordinate of the output tensor.

        :attr:`tensor_stride` (torch.IntTensor): the D-dimensional vector
        defining the stride between tensor elements.

    """
    if min_coordinate is not None:
        ## assert isinstance(min_coordinate, torch.IntTensor)
        assert isinstance(min_coordinate, torch.IntTensor) or isinstance(min_coordinate, torch.cuda.IntTensor)
        assert min_coordinate.numel() == SparseTensor._D
    if shape is not None:
        assert isinstance(shape, torch.Size)
        assert len(shape) == SparseTensor._D + 2  # batch and channel
        if shape[1] != SparseTensor._F.size(1):
            shape = torch.Size([shape[0], SparseTensor._F.size(1), *[s for s in shape[2:]]])

    # Use int tensor for all operations
    ## tensor_stride = torch.IntTensor(SparseTensor.tensor_stride)
    tensor_stride = torch.IntTensor(SparseTensor.tensor_stride).to(SparseTensor.coordinates.device)
    if not isinstance(min_coordinate, torch.IntTensor):
        tensor_stride = tensor_stride.cuda(SparseTensor.F.device)

    # New coordinates
    batch_indices = SparseTensor.C[:, 0]

    # TODO, batch first
    if min_coordinate is None:
        min_coordinate, _ = SparseTensor.C.min(0, keepdim=True)
        min_coordinate = min_coordinate[:, 1:]
        coords = SparseTensor.C[:, 1:] - min_coordinate
    elif isinstance(min_coordinate, int) and min_coordinate == 0:
        coords = SparseTensor.C[:, 1:]
    else:
        if min_coordinate.ndim == 1:
            min_coordinate = min_coordinate.unsqueeze(0)
        coords = SparseTensor.C[:, 1:] - min_coordinate

    assert (
                   min_coordinate % tensor_stride
           ).sum() == 0, "The minimum coordinates must be divisible by the tensor stride."

    if coords.ndim == 1:
        coords = coords.unsqueeze(1)

    # return the contracted tensor
    if contract_stride:
        coords = coords // tensor_stride

    nchannels = SparseTensor.F.size(1)
    if shape is None:
        size = coords.max(0)[0] + 1
        ## shape = torch.Size([batch_indices.max() + 1, nchannels, *size.numpy()])
        shape = torch.Size([batch_indices.max() + 1, nchannels, *size.cpu().numpy()])

    ## dense_F = torch.zeros(shape, dtype=SparseTensor.F.dtype, device=SparseTensor.F.device)
    dense_F = torch.full(shape, dtype=SparseTensor.F.dtype, device=SparseTensor.F.device, fill_value=default_value)

    tcoords = coords.t().long()
    batch_indices = batch_indices.long()
    exec(
        "dense_F[batch_indices, :, "
        + ", ".join([f"tcoords[{i}]" for i in range(len(tcoords))])
        + "] = SparseTensor.F"
    )

    ## tensor_stride = torch.IntTensor(SparseTensor.tensor_stride)
    tensor_stride = torch.IntTensor(SparseTensor.tensor_stride).to(SparseTensor.coordinates.device)
    return dense_F, min_coordinate, tensor_stride