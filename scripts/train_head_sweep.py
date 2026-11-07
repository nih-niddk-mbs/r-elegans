"""Fit a standalone stationary head-sweep policy through all 302 neurons."""
if __package__:
    from .train_steering import main
else:
    from train_steering import main

if __name__=='__main__':
    raise SystemExit(main(skill='head-sweep'))
