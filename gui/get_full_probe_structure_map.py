import json
import lungmap_utils
import gui.utils as gui_utils


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)

        return json.JSONEncoder.default(self, obj)


ontology = gui_utils.onto
probes = lungmap_utils.client.get_probes()

probe_structure_map = gui_utils.get_probe_structure_map(ontology, probes)

f = open('resources/probe_structure_map.json', 'w')
json.dump(probe_structure_map, f, cls=SetEncoder, indent=2)
f.close()
