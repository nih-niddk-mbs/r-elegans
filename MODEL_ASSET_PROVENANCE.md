# Bundled model asset provenance

`r_elegans/assets/runtime_model_v1.json` is a compact, project-generated
runtime checkpoint. It exists so the installed package can execute without
downloading scientific source files. It contains only canonical identifiers,
sparse derived topology/count arrays, and learned parameters; it does not
contain publication text, figures, spreadsheets, electrophysiology traces, or
generated trajectories.

Runtime asset SHA-256:
`012d1267f01c7a2df50af5cfb5d3f703d0af4295c20190af4fc47011984eb064`.
Single-compartment asset SHA-256:
`eaf247448a041782defbb4697ace4f4b4ab7a00ee988765a92c8cb1b4a33c545`.

## Anatomical topology

- Neuron-to-neuron chemical and gap-junction topology: Cook et al., “Whole-animal
  connectomes of both *Caenorhabditis elegans* sexes,” *Nature* 571, 63–71
  (2019), DOI: `10.1038/s41586-019-1352-7`.
- Neuron-to-body-wall-muscle topology: the hermaphrodite chemical-connectivity
  matrix from the same Cook et al. source.
- Neuromuscular polarity: Wang et al., “A neurotransmitter atlas of
  *C. elegans* males and hermaphrodites,” *eLife* (2024), DOI:
  `10.7554/eLife.95402`. ACh-only neurons are treated as excitatory, GABA-only
  neurons as inhibitory, and ambiguous classes remain unsigned.

All neural matrices use `[postsynaptic, presynaptic]` indexing. The bundled
topology contains 302 neurons, 3,709 directed chemical edges, 1,093 undirected
gap-junction pairs, and 956 neuron-to-muscle edges. Two unequal entries in the
source's nominally symmetric gap-junction sheet are resolved with the larger
pairwise count, and this transformation is recorded in the asset metadata.

## Electrophysiology parameters

`r_elegans/assets/single_compartment_v1.json` contains the project’s parameter
transcription for AWCON and RMD from Nicoletti et al., “Biophysical modeling of
*C. elegans* neurons,” *PLOS ONE* (2019), DOI:
`10.1371/journal.pone.0218738`, published under CC BY 4.0.

## Scope and licensing

The repository's MIT license applies to project code and project-generated
learned parameters. Source publications and their supplementary data retain
their respective rights and attribution requirements. The raw Cook, Wang,
Nicoletti, and validation artifacts are deliberately not redistributed here.

The checkpoint does not make the recurrent network biologically calibrated.
Its neuron-to-neuron graphs are anatomical topology, while the current working
chemotaxis demo uses a supervised command/phase-to-voltage motor teacher. Most
single-neuron conductances and recurrent synaptic strengths remain to be fit.
