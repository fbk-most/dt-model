import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import pandas as pd
import numpy as np
import math
import random

from dt_model import ContextVariable, PresenceVariable, Constraint, Ensemble, Model, Index, ConstIndex, UniformDistIndex, LognormDistIndex, TriangDistIndex, SymIndex

class VizSituation:    
    """
    Class represents contextual situation
    """

    def __init__(
        self,
        name: str,
        data
    ):
        self.name = name
        self.data = data

class VizModel:
    """
    Class represents visualization model data
    """

    def __init__(
        self,
        x: PresenceVariable,
        y: PresenceVariable,
        model: Model,
        situations: list[VizSituation]
    ):
        self.name = model.name
        self.x = x
        self.y = y
        self.model = model
        self.situations = situations
        self.situations_data = {} 
        self.kpis = {}
        self._build()


    def _build(self):
        xl = self.x.name
        yl = self.y.name
        ensemble_size = 20
        target_presence_samples = 200
        (x_max, y_max) = (10000, 10000)
        (x_sample, y_sample) = (100, 100)

        for ctx in self.situations:
            situation = ctx.data
            ensemble = Ensemble(self.model, situation, cv_ensemble_size=ensemble_size)
            xx = np.linspace(0, x_max, x_sample + 1)
            yy = np.linspace(0, y_max, y_sample + 1)
            xxg, yyg = np.meshgrid(xx, yy)
            zz = self.model.evaluate({self.x: xx, self.y: yy}, ensemble)
        
            sample_x = [sample for c in ensemble for sample in
                           self.x.sample(cvs=c[1], nr=max(1,round(c[0]*target_presence_samples)))]
            sample_y = [sample for c in ensemble for sample in
                                self.y.sample(cvs=c[1], nr=max(1,round(c[0]*target_presence_samples)))]

            # TODO: move elsewhere, it cannot be computed this way...
            area = self.model.compute_sustainable_area()
            index = self.model.compute_sustainability_index(list(zip(sample_x, sample_y)))
            indexes = self.model.compute_sustainability_index_per_constraint(list(zip(sample_x, sample_y)))
            critical = min(indexes, key=indexes.get)
            modals = self.model.compute_modal_line_per_constraint()

            self.kpis[ctx.name] = {
                "Area": f"{area / 10e6:.2f} kp$^2$",
                "Sustainability": f"{index * 100:.2f}%",
                "Critical": f"{critical.capacity.name} ({indexes[critical] * 100:.2f}%)"
            }

            # TODO: change, terrible
            self.situations_data[ctx.name] = {
                "xx": xxg,
                "yy": yyg,
                "zz": zz,
                "sample_x": sample_x,
                "sample_y": sample_y,
                "x_max": x_max,
                "y_max": y_max,
                "modals": modals
            }
            self.model.reset()
        
    
    def viz(self, ctx_name):        
        calculated = self.situations_data[ctx_name] 

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.pcolormesh(calculated["xx"], calculated["yy"], calculated["zz"], cmap='coolwarm_r', vmin=0.0, vmax=1.0)
        for modal in calculated["modals"].values():
            ax.plot(*modal, color='black', linewidth=2)
        ax.scatter(calculated["sample_y"], calculated["sample_x"], color='gainsboro', edgecolors='black')
        ax.set_title(self.name, fontsize=10)
        ax.set_xlim(left=0, right=calculated["x_max"])
        ax.set_ylim(bottom=0, top=calculated["y_max"])

        fig.colorbar(mappable=ScalarMappable(Normalize(0, 1), cmap='coolwarm_r'), ax=ax)
        fig.supxlabel(self.x.name)
        fig.supylabel(self.y.name)

        return fig

    def rename(self, name: str):
        old = self.name
        self.name = name
        self.model.name = name
        for idx in self.model.indexes:
            idx.name = idx.name.replace(f"({old})", f"({name})")
        for idx in self.model.capacities:
            idx.name = idx.name.replace(f"({old})", f"({name})")
        


    def clone(self, model_name: str, values: dict = {}):
        change_indexes = {}
        change_capacities = {}
        
        for idx in self.model.indexes:
            if idx.ref_name in values:
                var_idx = index_variation(idx, values[idx.ref_name], model_name)
                if var_idx:
                    change_indexes[idx] = var_idx
        for idx in self.model.capacities:
            if idx.ref_name in values:
                var_idx = index_variation(idx, values[idx.ref_name], model_name)
                if var_idx:
                    change_capacities[idx] = var_idx

        vm = None
        if change_capacities or change_indexes:
            nu_model = self.model.variation(model_name, change_indexes=change_indexes, change_capacities=change_capacities)
            vm = VizModel(self.x, self.y, nu_model, self.situations, list(change_indexes.values()) + list(change_capacities.values()))
        else: 
            nu_model = self.model.variation(model_name)
            vm = VizModel(self.x, self.y, nu_model, self.situations)

        return vm

    def update(self, values: dict = {}):
        change_indexes = {}
        change_capacities = {}

        for idx in self.model.indexes:
            if idx.ref_name in values:
                var_idx = index_variation(idx, values[idx.ref_name], self.name, True)
                if var_idx:
                    change_indexes[idx] = var_idx
        for idx in self.model.capacities:
            if idx.ref_name in values:
                var_idx = index_variation(idx, values[idx.ref_name], self.name, True)
                if var_idx:
                    change_capacities[idx] = var_idx

        if change_capacities or change_indexes:
            self.situations_data = {} 
            self.model = self.model.variation(self.name, change_indexes=change_indexes, change_capacities=change_capacities)
            self.kpis = {}
            self._build()

    def diff(self, other):
        res = []
        a = {idx.ref_name: idx for idx in self.model.indexes + self.model.capacities}
        b = {idx.ref_name: idx for idx in other.model.indexes + other.model.capacities}
        for k in a:
            if not is_equal_index(a[k], b[k]): res.append(a[k])
        return res

class VizApp:
    """
    Class implementing the whole modelling application with models, situations, and variables
    """

    def __init__(self, 
                 base_model: Model,
                 situations: dict, 
                 x: PresenceVariable, 
                 x_max: int, 
                 y: PresenceVariable, 
                 y_max: int):
        self.base_model = base_model
        self.situations = []
        for k in situations: self.situations.append(VizSituation(k, situations[k]))
        self.x = x
        self.x_max = x_max
        self.y = y
        self.y_max = y_max
        self.vis_models = [VizModel(x, y, base_model, self.situations)]
        self.__build_groups()
                     

    def __build_groups(self):
        group_dict = {}
        for c in (self.base_model.capacities + self.base_model.indexes):
            group = c.group if c.group is not None else "general"
            if group not in group_dict: group_dict[group] = {"id": group, "label": group, "parameters": [], "constraints": []}
            group_dict[group]["parameters"].append(c)
        for c in self.base_model.constraints:
            group = c.group if c.group is not None else "general"
            if group not in group_dict: group_dict[group] = {"id": group, "label": group, "parameters": [], "constraints": []}
            group_dict[group]["constraints"].append(c)
        self.groups = []
        for g in group_dict: self.groups.append(group_dict[g])

    def remove_model(self, model_name: str):
        self.vis_models = [m for m in self.vis_models if m.name != model_name]

    def add_model(self, name: str, model: VizModel):
        model.rename(name)
        self.vis_models.append(model)
        
def __value_to_min(v):
    if v < 0: return v * 2.0
    if v > 0: return 0.0
    else: return -1000.0

def __value_to_max(v):
    if v > 0: return v * 2.0
    if v < 0: return 0.0
    else: return 1000.0 


def index_to_widget(st, idx: Index, on_change=None):
    if isinstance(idx, ConstIndex):
        return st.slider(idx.name, __value_to_min(idx.v), __value_to_max(idx.v), idx.v, on_change=on_change, key=idx.name)
    if isinstance(idx, UniformDistIndex):
        return st.slider(idx.name, __value_to_min(idx.loc), __value_to_max(idx.loc + idx.scale), (idx.loc, idx.loc + idx.scale), on_change=on_change, key=idx.name)
    if isinstance(idx, LognormDistIndex):
        return st.slider(idx.name, __value_to_min(idx.loc), __value_to_max(idx.loc + idx.scale), (idx.loc, idx.loc + idx.scale), on_change=on_change, key=idx.name)
    if isinstance(idx, TriangDistIndex):
        return st.slider(idx.name, __value_to_min(idx.loc), __value_to_max(idx.loc + idx.scale), (idx.loc, idx.loc + idx.scale), on_change=on_change, key=idx.name)
    return None

def index_variation(idx, value, model_name, in_place: bool = False):
    ref_name = idx.ref_name
    var_name = f"{ref_name} ({model_name})"
    if isinstance(idx, ConstIndex):
        if value != idx.v: 
            if in_place:
                idx.v = value
                return idx
            else: return ConstIndex(var_name, value, idx.group, ref_name=ref_name)
    if isinstance(idx, UniformDistIndex):
        (f, t) = value
        if f != idx.loc or t != idx.loc + idx.scale: 
            if in_place:
                idx.loc = f
                idx.scale = t - f
                return idx
            else: return UniformDistIndex(var_name, f, t - f, idx.group, ref_name=ref_name)
    if isinstance(idx, LognormDistIndex):
        (f, t) = value
        if f != idx.loc or t != idx.loc + idx.scale: 
            if in_place:
                idx.loc = f
                idx.scale = t - f
                return idx
            else: return LognormDistIndex(var_name, f, t - f, idx.s, idx.group, ref_name=ref_name)
    if isinstance(idx, TriangDistIndex):
        (f, t) = value
        if f != idx.loc or t != idx.loc + idx.scale: 
            if in_place:
                idx.loc = f
                idx.scale = t - f
                return idx
            else: return TriangDistIndex(var_name, f, t - f, idx.c, idx.group, ref_name=ref_name)
    return None

# TODO move to index
def is_equal_index(idx1, idx2):
    if type(idx1) != type(idx2): return False
    if isinstance(idx1, ConstIndex):
        if idx1.value == idx2.value: return True
    if isinstance(idx1, UniformDistIndex) or isinstance(idx1, LognormDistIndex) or isinstance(idx1, TriangDistIndex):
        if idx1.loc == idx2.loc and idx1.scale == idx2.scale: return True

    if isinstance(idx1, SymIndex): return True
    return False
