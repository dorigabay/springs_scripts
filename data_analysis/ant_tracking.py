import os
import numpy as np
import scipy.io as sio
from scipy.ndimage.morphology import binary_closing


def first_tracking_calculation(directory,sub_dirs_names):
    complete_ants_centers_x = np.array([])
    complete_ants_centers_y = np.array([])
    for count,sub_dir in enumerate(sub_dirs_names):
        path = os.path.join(directory, sub_dir, "raw_analysis")
        print(f"loading from: {path}")
        ants_centers_x = np.loadtxt(os.path.join(path, "ants_centers_x.csv"), delimiter=",")
        ants_centers_y = np.loadtxt(os.path.join(path, "ants_centers_y.csv"), delimiter=",")
        if count == 0:
            complete_ants_centers_x = ants_centers_x
            complete_ants_centers_y = ants_centers_y
        else:
            complete_ants_centers_x = np.concatenate((complete_ants_centers_x, ants_centers_x))
            complete_ants_centers_y = np.concatenate((complete_ants_centers_y, ants_centers_y))
    ants_centers = np.stack((complete_ants_centers_x, complete_ants_centers_y), axis=2)
    ants_centers_mat = np.zeros((ants_centers.shape[0], 1), dtype=np.object)
    for i in range(ants_centers.shape[0]):
        ants_centers_mat[i, 0] = ants_centers[i, :, :]
    output_dir = os.path.join(directory, "ant_tracking")
    os.makedirs(output_dir, exist_ok=True)
    print(f"saving to: {output_dir}")
    sio.savemat(os.path.join(output_dir, "ants_centers.mat"), {"ants_centers": ants_centers_mat})
    matlab_script_path = "Z:\\Dor_Gabay\\ThesisProject\\scripts\\munkres_tracker\\"
    os.chdir(matlab_script_path)
    os.system("PATH=$PATH:'C:\Program Files\MATLAB\R2022a\bin'")
    execution_string = f"matlab -r ""ants_tracking('" + output_dir + "\\')"""
    os.system(execution_string)


from line_profiler_pycharm import profile

# @profile
def correct_tracked_ants(directory, tracked_ants, frame_size=(1920, 1080)):
    "This function removes all new labels that were create inside the frame, and not on the boundries (10% of the frame)"
    def test_if_in_boundries(coords):
        if coords[0] > frame_size[1] * 0.1 and coords[0] < frame_size[0] - frame_size[1] * 0.1 and coords[1] > frame_size[1] * 0.1 and coords[1] < frame_size[1] * 0.9:
            return True
        else:
            return False
    frame_size = (1920, 1080)
    num_of_frames = tracked_ants.shape[2]

    frames_arange = np.repeat(np.arange(tracked_ants.shape[2])[np.newaxis, :], tracked_ants.shape[0], axis=0)
    ants_arange = np.repeat(np.arange(tracked_ants.shape[0])[np.newaxis, :], tracked_ants.shape[2], axis=0).T
    tracked_ants[np.isnan(tracked_ants)] = 0

    appeared_labels = np.array([])
    appeared_labels_idx = []
    appeared_last_coords = np.array([]).reshape(0, 2)
    disappeared_labels = []
    disappeared_labels_partners = []
    disappeared_labels_partners_relative_distance = []
    disappeared_labels_partners_idx = []
    for frame in range(1,num_of_frames-1):
        print("\r frame: ", frame, end="")
        if frame+20>=num_of_frames:
            break

        if frame >= 11: min_frame = frame - 10
        else: min_frame = 1
        max_frame = frame + 15000


        labels_in_window = set(tracked_ants[:,2,min_frame:frame+20].flatten())
        labels_scanned_for_appearing = set(tracked_ants[:,2,min_frame-1].flatten())
        labels_now = set(tracked_ants[:, 2, frame])
        labels_before = set(tracked_ants[:, 2, frame - 1])
        appearing_labels = labels_in_window.difference(labels_scanned_for_appearing)
        disappearing_labels = labels_before.difference(labels_now)

        appeared_not_collected = appearing_labels.difference(set(appeared_labels))
        for label in appeared_not_collected:
            label_idx = np.isin(tracked_ants[:, 2, min_frame:max_frame], label)
            label_idx = (ants_arange[:, min_frame:max_frame][label_idx],frames_arange[:, min_frame:max_frame][label_idx])
            coords = tracked_ants[label_idx[0], 0:2, label_idx[1]]
            first_appear = np.argmin(label_idx[1])
            if test_if_in_boundries(coords[first_appear, :]):
                appeared_labels = np.append(appeared_labels,label)
                median_coords = coords[first_appear,:].reshape(1,2)
                appeared_last_coords = np.concatenate((appeared_last_coords, median_coords), axis=0)
                appeared_labels_idx.append(label_idx)

        currently_appearing = np.array([x for x in range(len(appeared_labels)) if appeared_labels[x] in appearing_labels])
        if len(currently_appearing)>0:
            currently_appearing_labels = appeared_labels[currently_appearing]
            currently_appearing_idx = [appeared_labels_idx[x] for x in currently_appearing]
            currently_appearing_last_coords = appeared_last_coords[currently_appearing]
            for label in disappearing_labels:
                coords = tracked_ants[tracked_ants[:, 2, frame-1]==label, 0:2, frame-1].flatten()
                if test_if_in_boundries(coords):
                    if currently_appearing_last_coords.shape[0] > 0:
                        distances = np.linalg.norm(currently_appearing_last_coords - coords, axis=1)
                        closest_appearing_label_argmin = np.argsort(distances)[0]
                        if label==currently_appearing_labels[closest_appearing_label_argmin] \
                                and len(distances)>1:
                            closest_appearing_label_argmin = np.argsort(distances)[1]
                        partner_label = currently_appearing_labels[closest_appearing_label_argmin]
                        distance_closest = distances[closest_appearing_label_argmin]
                        idx_closest_appearing_label = currently_appearing_idx[closest_appearing_label_argmin]
                        if distance_closest<300:
                            disappeared_labels.append(label)
                            disappeared_labels_partners.append(partner_label)
                            disappeared_labels_partners_relative_distance.append(distance_closest)
                            disappeared_labels_partners_idx.append(idx_closest_appearing_label)
    for i in range(len(disappeared_labels)):
        idx = disappeared_labels_partners_idx[i]
        tracked_ants[idx[0], 2, idx[1]] = disappeared_labels[i]
        if disappeared_labels_partners[i] in disappeared_labels:
            disappeared_labels[disappeared_labels.index(disappeared_labels_partners[i])] = disappeared_labels[i]
        duplicated_idxs = np.arange(len(disappeared_labels_partners))[np.array(disappeared_labels_partners)==disappeared_labels_partners[i]]
        duplicated_labels = [disappeared_labels[x] for x in duplicated_idxs]
        disappeared_labels = [np.min(duplicated_labels) if x in duplicated_labels else x for x in disappeared_labels]
    for unique_label in np.unique(tracked_ants[:, 2, :]):
        occurrences = np.count_nonzero(tracked_ants[:, 2, :] == unique_label)
        if occurrences < 20:
            idx = np.where(tracked_ants[:, 2, :] == unique_label)
            tracked_ants[idx[0], 2, idx[1]] = 0
    sio.savemat(os.path.join(os.path.join(directory), "tracking_data_corrected.mat"),{"tracked_blobs_matrix": tracked_ants})
    return tracked_ants


def assign_ants_to_springs(tracked_ants,ants_attached_labels,num_of_frames):
    ants_assigned_to_springs = np.zeros((num_of_frames, np.nanmax(tracked_ants[:,2,:]).astype(np.int32)))
    for i in range(num_of_frames):
        labels = tracked_ants[~np.isnan(ants_attached_labels[i]),2,i]
        labels = labels[~np.isnan(labels)].astype(int)
        springs = ants_attached_labels[i,~np.isnan(ants_attached_labels[i])].astype(int)
        ants_assigned_to_springs[i,labels-1] = springs+1
    interpolate_assigned_ants(ants_assigned_to_springs,num_of_frames)
    return ants_assigned_to_springs


def interpolate_assigned_ants(ants_assigned_to_springs,num_of_frames,max_gap=30):
    aranged_frames = np.arange(num_of_frames)
    for ant in range(ants_assigned_to_springs.shape[1]):
        array = ants_assigned_to_springs[:,ant]
        closing_bool = binary_closing(array.astype(bool).reshape(1,num_of_frames), np.ones((1, max_gap)))
        closing_bool = closing_bool.reshape(closing_bool.shape[1])
        xp = aranged_frames[~closing_bool+(array!=0)]
        fp = array[xp]
        x = aranged_frames[~closing_bool+(array==0)]
        ants_assigned_to_springs[x,ant] = np.round(np.interp(x, xp, fp))


def remove_artificial_cases(N_ants_around_springs,ants_assigned_to_springs):
    print("Removing artificial cases...")
    unreal_ants_attachments = np.full(N_ants_around_springs.shape, np.nan)
    switch_attachments = np.full(N_ants_around_springs.shape, np.nan)
    unreal_detachments = np.full(N_ants_around_springs.shape, np.nan)
    for ant in range(ants_assigned_to_springs.shape[1]):
        attachment = ants_assigned_to_springs[:,ant]
        if not np.all(attachment == 0):
            first_attachment = np.where(attachment != 0)[0][0]
            for count, spring in enumerate(np.unique(attachment)[1:]):
                frames = np.where(attachment==spring)[0]
                sum_assign = np.sum(ants_assigned_to_springs[frames[0]:frames[-1]+3,:] == spring, axis=1)
                real_sum = N_ants_around_springs[frames[0]:frames[-1]+3,int(spring-1)]
                if frames[0] == first_attachment and np.all(sum_assign[0:2] != real_sum[0:2]):
                    unreal_ants_attachments[frames,int(spring-1)] = ant
                elif np.all(sum_assign[0:2] != real_sum[0:2]):
                    switch_attachments[frames,int(spring-1)] = ant
                else:
                    if np.all(sum_assign[-1] != real_sum[-1]):
                        unreal_detachments[frames[-1], int(spring-1)] = ant
    for frame in range(ants_assigned_to_springs.shape[0]):
        detach_ants = np.unique(unreal_detachments[frame,:])
        for detach_ant in detach_ants[~np.isnan(detach_ants)]:
            spring = np.where(unreal_detachments[frame,:] == detach_ant)[0][0]
            assign_ant = unreal_ants_attachments[frame,spring]
            if not np.isnan(assign_ant):
                ants_assigned_to_springs[unreal_ants_attachments[:,spring]==assign_ant,int(detach_ant)] = spring+1
                ants_assigned_to_springs[unreal_ants_attachments[:,spring]==assign_ant,int(assign_ant)] = 0
        switch_ants = np.unique(switch_attachments[frame, :])
        for switch_ant in switch_ants[~np.isnan(switch_ants)]:
            spring = np.where(switch_attachments[frame,:] == switch_ant)[0][0]
            switch_from_spring = np.where(switch_attachments[frame-1,:] == switch_ant)[0]
            switch_from_spring = switch_from_spring[switch_from_spring != spring]
            if len(switch_from_spring) > 0:
                switch_frames = np.where(switch_attachments[:,spring] == switch_ant)[0]
                ants_assigned_to_springs[switch_frames, int(switch_ant)] = switch_from_spring[0]
    ants_assigned_to_springs = ants_assigned_to_springs[:,:-1]
    return ants_assigned_to_springs


def tracking_correction(directory,sub_dirs_names):
    tracked_ants = sio.loadmat(os.path.join(directory, "ant_tracking", "tracking_data.mat"))["tracked_blobs_matrix"]
    print("Correcting tracking...")
    tracked_ants = correct_tracked_ants(os.path.join(directory, "ant_tracking"), tracked_ants)
    # tracked_ants = sio.loadmat(os.path.join(directory, "ant_tracking", "tracking_data_corrected.mat"))["tracked_blobs_matrix"]
    N_ants_around_springs = np.load(os.path.join(directory,"two_vars_post_processing", "N_ants_around_springs.npz"))["arr_0"]
    last_frame = 0
    for count, sub_dir in enumerate(sub_dirs_names):
        print(f"Processing {sub_dir}...")
        path_raw_analysis = os.path.join(directory, sub_dir, "raw_analysis")
        path_post_processing = os.path.join(directory, sub_dir, "two_vars_post_processing")
        ants_attached_labels = np.loadtxt(os.path.join(path_raw_analysis, "ants_attached_labels.csv"), delimiter=",")
        ants_assigned_to_springs = assign_ants_to_springs(
                                            tracked_ants[:, :, last_frame:last_frame+ants_attached_labels.shape[0]],
                                            ants_attached_labels,
                                            ants_attached_labels.shape[0])
        ants_assigned_to_springs = remove_artificial_cases(
                        N_ants_around_springs[last_frame:last_frame+ants_attached_labels.shape[0], :], ants_assigned_to_springs)
        np.savez_compressed(os.path.join(directory,"two_vars_post_processing", "ants_assigned_to_springs.npz"), ants_assigned_to_springs)
        last_frame += ants_attached_labels.shape[0]
    print("Done!")

if __name__=="__main__":
    directory = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\15.9.22\\plus0.3mm_force\\"
    sub_dirs_names = [f"S528000{i}" for i in [8, 9]]
    first_tracking_calculation(directory,sub_dirs_names)
    # # wait for 3 hours
    from time import sleep
    sleep(2*60*60)
    tracking_correction(directory, sub_dirs_names)
    # sub_dirs_names = [f"S528000{i}" for i in [3, 4, 5, 6, 7]]
    # first_tracking_calculation(directory,sub_dirs_names)
    # # # wait for 3 hours
    # from time import sleep
    # sleep(4*60*60)
    # tracking_correction(directory, sub_dirs_names)
