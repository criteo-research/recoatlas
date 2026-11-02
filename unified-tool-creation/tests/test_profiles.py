from tools_builder.profiles import available_profiles, load_profile


def test_all_paper_profiles_are_valid():
    assert available_profiles() == [
        "amazon-electronics",
        "amazon-musical-instruments",
        "amazon-video-games",
        "hm-clothing",
    ]
    for name in available_profiles():
        assert load_profile(name)["profile"] == name


def test_profiles_preserve_domain_specific_complementarity():
    amazon = load_profile("amazon-musical-instruments")
    clothing = load_profile("hm-clothing")
    assert "MLP" in amazon["complementarity_architecture"]
    assert "residual" in clothing["complementarity_architecture"]
    assert clothing["selected_complementarity_epoch"] == 2
