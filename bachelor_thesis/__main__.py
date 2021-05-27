import skeltorch
from .data import BachelorThesisData
from .runner import BachelorThesisRunner

skeltorch.Skeltorch(BachelorThesisData(), BachelorThesisRunner()).run()
