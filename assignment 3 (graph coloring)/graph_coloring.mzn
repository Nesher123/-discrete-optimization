% Input parameters (values for these passed from script calling it)
int: NODE_COUNT;
int: EDGE_COUNT;
array[1..EDGE_COUNT,1..2] of int: EDGES;

% Decision Variable: the solver will find the values of these
array[0..NODE_COUNT-1] of var 1..NODE_COUNT: colors;

% Our Constraints
constraint forall(e in 1..EDGE_COUNT)(colors[EDGES[e,1]] != colors[EDGES[e,2]]);

% Our Objective Function
solve minimize max(colors);

% formatted output
output[show(colors)];
