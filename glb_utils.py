#!/usr/bin/env python3

import argparse
from typing import List, Dict

from magnum import *
from magnum import math, meshtools, scenetools, trade

# Scene conversion plugins need access to image conversion plugins
importer_manager = trade.ImporterManager()

# Import / export plugins
importer = importer_manager.load_and_instantiate('AnySceneImporter')

# if args.quiet:
#     importer.flags |= trade.ImporterFlags.QUIET
# if args.verbose:
#     importer.flags |= trade.ImporterFlags.VERBOSE

def get_total_vertex_count(filepath):

  importer.open_file(filepath)
  # scene = importer.scene(0)
  # Go through all meshes
  total_triangle_count = 0
  for i in range(importer.mesh_count):
      mesh = importer.mesh(i)

      if not mesh.is_indexed:
          assert False, "i didn't bother with index-less variant for the heuristics, sorry"
      if mesh.primitive != MeshPrimitive.TRIANGLES:
          assert False, "i didn't bother with non-triangle meshes either, sorry"

      total_triangle_count += mesh.index_count//3

  return total_triangle_count
