# Observers

Observers collect read-only evidence such as operator timelines, tensor
metadata, finite/non-finite statistics, cache relations, and recovery status.
They publish evidence events but do not participate in injection selection.

MindStudio Probe integration belongs here as an observer/orchestrator. It must
not become a separate injection backend merely because enabling it changes the
runtime call surface.
