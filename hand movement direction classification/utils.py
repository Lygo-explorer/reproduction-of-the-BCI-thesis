from pathlib import Path
import re
import numpy as np
import mne
from scipy.io import loadmat
from scipy.signal import cheby1, sosfiltfilt

BCICIV2A_LABELS = {
    "769": 0,  # left hand
    "770": 1,  # right hand
    "771": 2,  # feet
    "772": 3,  # tongue
}
BCICIV2A_CHANNELS = (
    'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2',
    'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3',
    'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4',
    'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11',
    'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz',
    'EEG-15', 'EEG-16', 'EOG-left', 'EOG-central', 'EOG-right'
)

BCICIV2B_LABELS = {
    "769": 0,  # left hand
    "770": 1,  # right hand
}
BCICIV_UNKNOWN_CUE_EVENTS = ("783", "768")

def _find_event_code(event_id, annotation_code):
    for key, value in event_id.items():
        if str(key).strip().split(".")[0] == annotation_code:
            return value
    return None


def _pick_motor_imagery_channels(raw, channels):
    if channels is None:
        raw.pick_types(eeg=True, eog=False, stim=False)
        return raw

    channel_names = []
    normalized = {_normalize_channel_name(name): name for name in raw.ch_names}

    for channel in channels:
        target = _normalize_channel_name(channel)
        match = normalized.get(target)

        if match is None:
            suffixes = (":" + target, "-" + target, "." + target)
            for normalized_name, original_name in normalized.items():
                if normalized_name.endswith(suffixes):
                    match = original_name
                    break

        if match is None:
            raise ValueError(
                f"Cannot find channel {channel!r}. Available channels: {raw.ch_names}"
            )

        channel_names.append(match)

    raw.pick_channels(channel_names)
    return raw


def _normalize_channel_name(channel_name):
    return channel_name.replace(" ", "").lower()


def _subject_from_gdf_name(gdf_file, subject_pattern):
    match = re.match(subject_pattern, gdf_file.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot parse subject id from file name: {gdf_file.name}")
    return int(match.group(1))


def _load_true_label(mat_file, n_classes):
    if not mat_file.exists():
        raise FileNotFoundError(f"True label file not found: {mat_file}")

    mat = loadmat(mat_file)
    if "classlabel" not in mat:
        available_keys = [key for key in mat.keys() if not key.startswith("__")]
        raise KeyError(
            f"'classlabel' was not found in {mat_file}. Available keys: {available_keys}"
        )

    label = np.asarray(mat["classlabel"]).reshape(-1).astype(np.int64)
    unique_label = set(np.unique(label))
    one_based_labels = set(range(1, n_classes + 1))
    zero_based_labels = set(range(n_classes))

    if unique_label <= one_based_labels:
        label = label - 1
    elif not unique_label <= zero_based_labels:
        raise ValueError(
            f"Unexpected labels in {mat_file}: {sorted(unique_label)}. "
            f"Expected labels are {sorted(one_based_labels)} or "
            f"{sorted(zero_based_labels)}."
        )

    return label


def load_bciciv2a(
        root_dir,
        tmin=0.0,
        tmax=4.0,
        channels=BCICIV2A_CHANNELS,
):
    """
    Parameters
    ----------
    root_dir : str
        Dataset 2a所在文件夹

    tmin : float
        Trial开始时间(s)

    tmax : float
        Trial结束时间(s)

    channels : tuple[str] or None
        要读取的EEG通道。默认读取2a的22个EEG通道；传入None则读取所有EEG通道。

    Returns
    -------
    train_data : ndarray
        训练集数据，来自*T.gdf，形状为(S, Ns_train, C, T)

    test_data : ndarray
        测试集数据，来自*E.gdf，形状为(S, Ns_test, C, T)

    train_label : ndarray
        训练集标签，形状为(S, Ns_train)，标签为0/1/2/3

    test_label : ndarray
        测试集标签，形状为(S, Ns_test)，标签为0/1/2/3
    """

    return _load_bciciv_gdf_dataset(
        root_dir=root_dir,
        class_event_labels=BCICIV2A_LABELS,
        subject_pattern=r"A(\d{2})[TE]\.gdf$",
        channels=channels,
        tmin=tmin,
        tmax=tmax,
    )


def load_bciciv2b(
        root_dir,
        tmin=0.0,
        tmax=4.0,
        channels=("C3", "Cz", "C4"),
):
    """
    Parameters
    ----------
    root_dir : str
        Dataset 2b所在文件夹

    tmin : float
        Trial开始时间(s)

    tmax : float
        Trial结束时间(s)

    channels : tuple[str]
        要读取的EEG通道，默认读取C3、Cz、C4

    Returns
    -------
    train_data : ndarray
        训练集数据，来自*T.gdf，形状为(S, Ns_train, C, T)

    test_data : ndarray
        测试集数据，来自*E.gdf，形状为(S, Ns_test, C, T)

    train_label : ndarray
        训练集标签，形状为(S, Ns_train)

    test_label : ndarray
        测试集标签，形状为(S, Ns_test)
    """

    return _load_bciciv_gdf_dataset(
        root_dir=root_dir,
        class_event_labels=BCICIV2B_LABELS,
        subject_pattern=r"B(\d{2})\d{2}[TE]\.gdf$",
        channels=channels,
        tmin=tmin,
        tmax=tmax,
    )


def _load_bciciv_gdf_dataset(
        root_dir,
        class_event_labels,
        subject_pattern,
        channels,
        tmin,
        tmax,
        train_pattern="*T.gdf",
        test_pattern="*E.gdf",
        true_label_folder="true_labels",
):
    root_dir = Path(root_dir)
    true_label_dir = root_dir / true_label_folder

    train_data, train_label = _load_bciciv_split(
        root_dir=root_dir,
        pattern=train_pattern,
        split_name="training",
        tmin=tmin,
        tmax=tmax,
        channels=channels,
        true_label_dir=None,
        class_event_labels=class_event_labels,
        subject_pattern=subject_pattern,
    )
    test_data, test_label = _load_bciciv_split(
        root_dir=root_dir,
        pattern=test_pattern,
        split_name="test",
        tmin=tmin,
        tmax=tmax,
        channels=channels,
        true_label_dir=true_label_dir,
        class_event_labels=class_event_labels,
        subject_pattern=subject_pattern,
    )

    return (
        train_data,
        test_data,
        train_label,
        test_label,
    )


def _load_bciciv_split(
        root_dir,
        pattern,
        split_name,
        tmin,
        tmax,
        channels,
        true_label_dir,
        class_event_labels,
        subject_pattern,
):
    data_by_subject = {}
    label_by_subject = {}

    gdf_files = sorted(root_dir.glob(pattern))

    print(f"Found {len(gdf_files)} {split_name} GDF files")

    for gdf_file in gdf_files:

        print(f"Loading {gdf_file.name}")

        raw = mne.io.read_raw_gdf(
            gdf_file,
            preload=True,
            verbose=False
        )

        # 仅保留EEG
        raw = _pick_motor_imagery_channels(raw, channels)

        events, event_id = mne.events_from_annotations(raw)

        selected_event_id = {}
        code_to_label = {}
        for annotation_code, label_value in class_event_labels.items():
            event_code = _find_event_code(event_id, annotation_code)
            if event_code is not None:
                selected_event_id[annotation_code] = event_code
                code_to_label[event_code] = label_value

        label_from_mat = None
        if len(selected_event_id) == len(class_event_labels):
            epoch_event_id = selected_event_id
        elif true_label_dir is not None:
            mat_file = true_label_dir / f"{gdf_file.stem}.mat"
            label_from_mat = _load_true_label(
                mat_file,
                n_classes=len(class_event_labels),
            )
            epoch_event_id = {}
            for annotation_code in BCICIV_UNKNOWN_CUE_EVENTS:
                event_code = _find_event_code(event_id, annotation_code)
                if event_code is not None:
                    epoch_event_id[annotation_code] = event_code
                    break

            if not epoch_event_id:
                raise RuntimeError(
                    f"{gdf_file.name} does not contain class labels "
                    f"{tuple(class_event_labels)} or unknown cue events "
                    f"{BCICIV_UNKNOWN_CUE_EVENTS}."
                )
        else:
            raise RuntimeError(
                f"{gdf_file.name} does not contain class labels "
                f"{tuple(class_event_labels)}."
            )

        epochs = mne.Epochs(
            raw,
            events,
            event_id=epoch_event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            verbose=False
        )

        X = epochs.get_data()   # (trial, channel, sample)

        y = epochs.events[:, -1]

        if label_from_mat is None:
            # 转换成0/1标签
            y = np.array([code_to_label[event_code] for event_code in y], dtype=np.int64)
        else:
            y = label_from_mat[:len(epochs)]

        subject = _subject_from_gdf_name(gdf_file, subject_pattern)
        data_by_subject.setdefault(subject, []).append(X)
        label_by_subject.setdefault(subject, []).append(y)

        print(f"  Loaded {X.shape[0]} trials, data shape {X.shape}")

    if not data_by_subject:
        raise RuntimeError(
            f"No {split_name} files matching {pattern!r} were found in {root_dir}."
        )

    subjects = sorted(data_by_subject)
    data_list = []
    label_list = []

    for subject in subjects:
        subject_data = np.concatenate(data_by_subject[subject], axis=0)
        subject_label = np.concatenate(label_by_subject[subject], axis=0)
        data_list.append(subject_data)
        label_list.append(subject_label)

    return data_list, label_list

def cheby1_bandpass(data,
                    lowcut,
                    highcut,
                    fs,
                    order=6,
                    ripple=1):

    nyquist = fs / 2

    sos = cheby1(N=order, rp=ripple,  # 通带波纹(dB)
                Wn=[lowcut/nyquist, highcut/nyquist], btype='bandpass', output='sos')
    processed_data = sosfiltfilt(sos, data, axis=-1)
    return processed_data.copy()

# data augmentation
def segmentation_reconstruction(
    X,
    y,
    n_segments=8,
    n_aug_per_class=None,
    include_original=True,
    random_state=None
):
    """
    Segmentation & Reconstruction (S&R)

    Parameters
    ----------
    X : np.ndarray
        Shape = (N, C, T)

    y : np.ndarray
        Shape = (N,)

    n_segments : int
        Number of segments (paper uses 8)

    n_aug_per_class : int or None
        Number of generated samples per class

        None:
            generate same number as original class

    include_original : bool
        Whether to keep original samples

    random_state : int or None
        Random seed

    Returns
    -------
    X_out : np.ndarray
        Shape = (N_new, C, T)

    y_out : np.ndarray
        Shape = (N_new,)
    """

    if random_state is not None:
        np.random.seed(random_state)

    classes = np.unique(y)

    X_aug_list = []
    y_aug_list = []

    if include_original:
        X_aug_list.append(X)
        y_aug_list.append(y)

    for cls in classes:

        X_cls = X[y == cls]

        n_trials, C, T = X_cls.shape

        seg_len = T // n_segments

        if n_aug_per_class is None:
            n_aug = n_trials
        else:
            n_aug = n_aug_per_class

        X_new = np.zeros(
            (n_aug, C, T),
            dtype=X.dtype
        )

        for i in range(n_aug):

            trial = np.zeros(
                (C, T),
                dtype=X.dtype
            )

            for seg in range(n_segments):

                trial_idx = np.random.randint(
                    0,
                    n_trials
                )

                start = seg * seg_len

                if seg == n_segments - 1:
                    end = T
                else:
                    end = (seg + 1) * seg_len

                trial[:, start:end] = (
                    X_cls[
                        trial_idx,
                        :,
                        start:end
                    ]
                )

            X_new[i] = trial

        y_new = np.full(
            n_aug,
            cls,
            dtype=y.dtype
        )

        X_aug_list.append(X_new)
        y_aug_list.append(y_new)

    X_out = np.concatenate(
        X_aug_list,
        axis=0
    )

    y_out = np.concatenate(
        y_aug_list,
        axis=0
    )

    return X_out, y_out

# mkdir with Path
def mkdir_with_suffix(path: Path) -> Path:

    if not path.exists():
        path.mkdir(parents=True)
        return path

    i = 1
    while True:
        new_path = path.parent / f"{path.name}_{i}"
        if not new_path.exists():
            new_path.mkdir(parents=True)
            return new_path
        i += 1
