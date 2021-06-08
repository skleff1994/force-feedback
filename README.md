# force-feedback
Force feedback for optimal control : preliminary study on the point-mass system. 4 approaches are explored:
 - "Augmented state" approach: the force is included in the state (assuming a spring-damper contact model)
 - "Observer" approach: the force is an output used in state estimation (assuming a spring-damper contact model)
 - "Inverse gain" approach: pseudo-invert the partial derivative of the contact force w.r.t. control input (assuming a rigid contact model)
 - "Actuation dynamics" approach: the actuation is modeled as a low-pass filter and filtered torque become part of the state 
