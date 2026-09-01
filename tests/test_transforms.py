import pickle

from PIL import Image

from aigc_detector.transforms import (
    JpegCompression,
    RandomSingleOfficialTransform,
    evaluation_inference_transform,
    evaluation_transform,
    official_conditions,
    training_transform,
)


def test_all_official_conditions_produce_normalized_tensor():
    image = Image.new("RGB", (64, 48), color=(100, 150, 200))
    for preprocess_mode in ("stretch", "short_side_crop"):
        for condition in official_conditions():
            output = evaluation_transform(32, condition, preprocess_mode=preprocess_mode)(image)
            assert tuple(output.shape) == (3, 32, 32), (preprocess_mode, condition.name)


def test_flip_inference_policy_returns_two_views():
    image = Image.new("RGB", (64, 48), color=(100, 150, 200))
    output = evaluation_inference_transform(
        32,
        preprocess_mode="short_side_crop",
        inference_policy="reference_flip_mean",
    )(image)
    assert tuple(output.shape) == (2, 3, 32, 32)


def test_condition_names_are_unique():
    names = [condition.name for condition in official_conditions()]
    assert len(names) == len(set(names))


def test_robust_training_profile_produces_tensor():
    image = Image.new("RGB", (64, 48), color=(100, 150, 200))
    transform = training_transform(32, "robust", preprocess_mode="short_side_crop")
    output = transform(image)
    assert tuple(output.shape) == (3, 32, 32)
    assert isinstance(transform.transforms[0], RandomSingleOfficialTransform)
    assert len(transform.transforms[0].transforms) == 19
    assert [type(item).__name__ for item in transform.transforms[1:]] == [
        "Resize",
        "CenterCrop",
        "ToImage",
        "ToDtype",
        "Normalize",
    ]


def test_transforms_are_pickle_safe_for_worker_processes():
    pickle.dumps(training_transform(32, "robust"))
    for preprocess_mode in ("stretch", "short_side_crop"):
        for condition in official_conditions():
            pickle.dumps(evaluation_transform(32, condition, preprocess_mode=preprocess_mode))
            pickle.dumps(
                evaluation_inference_transform(
                    32,
                    condition,
                    preprocess_mode=preprocess_mode,
                    inference_policy="reference_flip_mean",
                )
            )


def test_label_independent_codec_normalization_is_applied_before_resize():
    image = Image.new("RGB", (64, 48), color=(100, 150, 200))
    train = training_transform(
        32,
        "robust",
        preprocess_mode="short_side_crop",
        codec_normalization="jpeg_q96",
    )
    assert isinstance(train.transforms[0], RandomSingleOfficialTransform)
    assert isinstance(train.transforms[1], JpegCompression)
    assert tuple(train(image).shape) == (3, 32, 32)

    evaluate = evaluation_transform(
        32,
        official_conditions()[1],
        codec_normalization="jpeg_q96",
    )
    assert isinstance(evaluate.transforms[0], JpegCompression)
    assert isinstance(evaluate.transforms[1], JpegCompression)
    assert tuple(evaluate(image).shape) == (3, 32, 32)


def test_unknown_codec_normalization_is_rejected():
    for factory in (evaluation_transform, training_transform):
        try:
            factory(32, codec_normalization="unknown")
        except ValueError as error:
            assert "unknown codec normalization" in str(error)
        else:
            raise AssertionError("unknown codec normalization was accepted")


def test_unknown_preprocessing_mode_is_rejected():
    try:
        evaluation_transform(32, preprocess_mode="unknown")
    except ValueError as error:
        assert "unknown preprocessing mode" in str(error)
    else:
        raise AssertionError("unknown preprocessing mode was accepted")
    try:
        training_transform(32, preprocess_mode="unknown")
    except ValueError as error:
        assert "unknown preprocessing mode" in str(error)
    else:
        raise AssertionError("unknown training preprocessing mode was accepted")
