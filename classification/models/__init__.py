from .pvmamba import PVMAMBA, Backbone_PVMAMBA
    
def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["pvmamba"]:
        model = PVMAMBA(
            in_chans=config.MODEL.PVMAMBA.IN_CHANS,
            patch_size=config.MODEL.PVMAMBA.PATCH_SIZE,
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.PVMAMBA.DEPTHS,
            embed_dim=config.MODEL.PVMAMBA.EMBED_DIM,
            d_state=config.MODEL.PVMAMBA.D_STATE,
            dt_init=config.MODEL.PVMAMBA.DT_INIT,
            mlp_ratio=config.MODEL.PVMAMBA.MLP_RATIO,
            token_mixer_types=config.MODEL.PVMAMBA.TOKEN_MIXER_TYPES,
            spatial_flag=config.MODEL.PVMAMBA.SPATIAL_FLAG,
            num_heads=config.MODEL.PVMAMBA.NUM_HEADS ,
            ssd_expansion=config.MODEL.PVMAMBA.SSD_EXPANSION,

            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,


        )
        return model

    return None
