trans = [0, 0.9, 0.05, 0.05; 0, 0.8, 0.15, 0.05; 0, 0, 0.7, 0.3; 0, 0, 0, 1];
emis = [1, 0; 0.75, 0.25; 0.5, 0.5; 0.1, 0.9];

num = 300;

seq = zeros(num,11);

for c = 1:num
    seq(c,:) = [hmmgenerate(10, trans, emis), 0];
end
seq = seq - 1;

probs = zeros(num,11);
for c = 1:num
    probs(c,:) = [0,0,0,0,0,0,0,0,0,0,-1];
end

csvwrite("observations_3state.csv", seq);
csvwrite("problems_3state.csv", probs);