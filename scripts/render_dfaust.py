
import argparse
import os
from os import path
import sys

from simple_3dviz import Mesh
from simple_3dviz.scenes import Scene
from simple_3dviz.utils import save_frame


def render_dfaust(scene, prev_renderable, seq, target):
    new_renderable = Mesh.from_file(seq)
    scene.remove(prev_renderable)
    scene.add(new_renderable)
    scene.render()
    save_frame(target, scene.frame)

    return new_renderable


def get_scene():
    scene = Scene((256, 256))
    scene.camera_position = (1, 1.5, 3)
    scene.camera_target = (0, 0.5, 0)
    scene.light = (1, 1.5, 3)
    #scene = Scene((512, 512))
    #scene.light = (1.0, 1.0, 3.0)
    #scene.camera_position = (0.4, 0.4, 1.4)
    #scene.camera_target = (0, 0.0, 0)
    scene.up_vector = (0, 1, 0)

    return scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render the D-FAUST dataset"
    )

    scene = get_scene()
    renderable = None
    for recording in sys.stdin:
        recording = recording.strip()
        mesh_base = path.join(recording, "mesh_seq")
        sequences = [
            path.join(mesh_base, seq)
            for seq in os.listdir(mesh_base)
            if seq.endswith("obj")
        ]
        print("Rendering {}".format(path.basename(recording)))
        renderings = path.join(recording, "renderings-downsampled")
        if not path.exists(renderings):
            os.mkdir(renderings)
        print("0 / {}".format(len(sequences)), end="")
        for i, seq in enumerate(sequences):
            target = seq.replace("obj", "png")
            target = target.replace("mesh_seq", "renderings-downsampled")
            renderable = render_dfaust(scene, renderable, seq, target)
            print("\r{} / {}".format(i, len(sequences)), end="")
        print()
