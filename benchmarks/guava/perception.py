"""A perfect segmenter for Guava's tools: masks rendered from MuJoCo's segmentation buffer.

Guava's release tools turn a mask into a point cloud and take its mean (get_position) or its
principal axes (grasp, align); its own ground-truth solver gives positions only, so it cannot
drive them. This keeps Guava's whole point-cloud pipeline and replaces only SAM3: same camera,
same resolution, exact masks.
"""
from __future__ import annotations

import numpy as np


def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


class SegMaskSolver:
    def __init__(self, env, aliases: dict[str, str]):
        self.env = env
        self.aliases = {_norm(k): v for k, v in aliases.items()}

    def _body(self, name: str) -> int:
        model = self.env.rob.sim.model
        sim_name = self.aliases.get(_norm(name), name)
        names = {model.body_id2name(i) for i in range(model._model.nbody)}
        for candidate in (sim_name, sim_name + "_main"):
            if candidate in names:
                return model.body_name2id(candidate)
        raise ValueError(f"unknown object {name!r}")

    def _geoms(self, name: str) -> np.ndarray:
        m = self.env.rob.sim.model._model
        root = self._body(name)
        bodies = {root}
        for b in range(m.nbody):  # every body below root
            p = b
            while p not in (0, root):
                p = int(m.body_parentid[p])
            if p == root:
                bodies.add(b)
        return np.array([g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies])

    def get_mask(self, name: str, rgb: np.ndarray) -> np.ndarray:
        import mujoco

        cam = self.env._cam_cfg or self.env.cfg.camera
        m, d = self.env.rob.sim.model._model, self.env.rob.sim.data._data
        h, w = rgb.shape[:2]
        if not hasattr(self, "_r") or self._r.width != w or self._r.height != h:
            self._r = mujoco.Renderer(m, height=h, width=w)
        self._r.enable_segmentation_rendering()
        self._r.update_scene(d, camera=cam.name)
        seg = self._r.render()  # (h, w, 2): object id, object type; top-down rows
        self._r.disable_segmentation_rendering()
        ids, types = seg[..., 0], seg[..., 1]
        return (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(ids, self._geoms(name))

    def get_position(self, name: str, **_) -> list[float]:
        return list(map(float, self.env.rob.sim.data.xpos[self._body(name)]))
