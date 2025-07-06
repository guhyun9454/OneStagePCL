def update_ood_hyperparams(args):
    """커맨드라인 인자를 기반으로 OOD 하이퍼파라미터를 업데이트"""
    
    from OODdetectors import ood_adapter as _oa

    # ENERGY
    _oa._DEFAULT_PARAMS.setdefault("ENERGY", {})["temperature"] = args.energy_temperature

    # GEN (공통: GEN, PRO_GEN)
    _oa._DEFAULT_PARAMS.setdefault("GEN", {})["gamma"] = args.gen_gamma
    _oa._DEFAULT_PARAMS.setdefault("GEN", {})["M"] = args.gen_M

    # PRO_GEN
    _oa._DEFAULT_PARAMS.setdefault("PRO_GEN", {})["gamma"] = args.gen_gamma
    _oa._DEFAULT_PARAMS.setdefault("PRO_GEN", {})["M"] = args.gen_M
    _oa._DEFAULT_PARAMS["PRO_GEN"]["noise_level"] = args.pro_gen_noise_level
    _oa._DEFAULT_PARAMS["PRO_GEN"]["gd_steps"] = args.pro_gen_gd_steps

    # PRO_MSP
    _oa._DEFAULT_PARAMS.setdefault("PRO_MSP", {})["temperature"] = args.pro_msp_temperature
    _oa._DEFAULT_PARAMS["PRO_MSP"]["noise_level"] = args.pro_msp_noise_level
    _oa._DEFAULT_PARAMS["PRO_MSP"]["gd_steps"] = args.pro_msp_gd_steps

    # PRO_MSP_T
    _oa._DEFAULT_PARAMS.setdefault("PRO_MSP_T", {})["temperature"] = args.pro_msp_t_temperature
    _oa._DEFAULT_PARAMS["PRO_MSP_T"]["noise_level"] = args.pro_msp_t_noise_level
    _oa._DEFAULT_PARAMS["PRO_MSP_T"]["gd_steps"] = args.pro_msp_t_gd_steps

    # PRO_ENT
    _oa._DEFAULT_PARAMS.setdefault("PRO_ENT", {})["noise_level"] = args.pro_ent_noise_level
    _oa._DEFAULT_PARAMS["PRO_ENT"]["gd_steps"] = args.pro_ent_gd_steps 