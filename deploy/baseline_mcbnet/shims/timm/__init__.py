"""timm stand-in: only timm.layers.DropPath / trunc_normal_ (and the timm.models.layers
alias), which Peter's MCBNet model file imports. Inference only: DropPath is identity in
eval mode, exactly as in timm."""
