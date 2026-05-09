// Long blobPop optimization — runs LHS in batches, writes results to file
const { KERNELS, optimize } = require('./optimize.js');

const result = optimize(['blobPop'], {
  samples: 200,
  time: 14,
  warmup: 8,
  kalmanIters: 5,
});

console.log('\n\nDONE');
