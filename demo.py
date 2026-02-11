from dovsg.controller import Controller
from dovsg.utils.utils import vis_depth
import argparse

def main(args):

    controller = Controller(
        step=0, 
        tags=args.tags, 
        interval=3, 
        resolution=0.01,
        occ_avoid_radius=0.2,
        save_memory=args.save_memory,
        debug=args.debug,
        venv_droid=args.venv_droid
    )

    if args.scanning_room:
        # data collection and pose estimation, if you have data, please don't use it
        # to avoid delte exist data
        controller.data_collection()

    if args.preprocess:
        controller.pose_estimation()

        # show droid-slam pose pointcloud
        controller.show_droidslam_pointcloud(use_inlier_mask=False, is_visualize=True)

        # transform droid-slam pose to floor base coord
        controller.transform_pose_with_floor(display_result=False)

        # use transformed pose train ace for relocalize
        controller.train_ace()

        # vis_depth(controller.recorder_dir)
   
        controller.show_pointcloud(is_visualize=True)

    # when first times, init scenario
    controller.get_view_dataset()
    controller.get_semantic_memory()
    controller.get_instances()
    controller.get_instance_scene_graph()
    controller.get_lightglue_features()

    # controller.show_pointcloud()
    """
        press "B" to show background
        press "C" to color by class
        press "R" to color by rgb
        press "F" to color by clip sim
        press "G" to toggle scene graph
        press "I" to color by instance
        press "O" to toggle bbox
        press "V" to save view params
    """
    controller.show_instances(
        controller.instance_objects, 
        clip_vis=True, 
        scene_graph=controller.instance_scene_graph, 
        show_background=True
    )
    
    if not args.scanning_room:
        tasks = controller.get_task_plan(description=args.task_description, change_level=args.task_scene_change_level)
        print(tasks)
        controller.run_tasks(tasks=tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='demo of dovsg.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--tags', type=str, default="room1", help='tags for scene.')
    parser.add_argument('--save_memory', type=bool, default=True, help='save each step memory.')

    parser.add_argument('--scanning_room', action='store_true', help='For hand camera to recorder scene.')
    parser.add_argument('--preprocess', action='store_true', help='preprocess scene.')
    parser.add_argument('--debug', action='store_true', help='For debug mode.')

    parser.add_argument('--task_scene_change_level', type=str, default="Minor Adjustment", 
                        choices=["Minor Adjustment", "Positional Shift", "Appearance"], help='scene change level.')
    parser.add_argument('--task_description', type=str, default="", help='your task description.')


    parser.add_argument('--venv_droid', type=str, default="", help='path to droidslam virtual environment')

    args = parser.parse_args()

    # args.task_scene_change_level = "Minor Adjustment"  # Minor Adjustment, Positional Shift, Appearance
    # args.task_description = "Please move the red pepper to the plate, then move the green pepper to plate."
    main(args)