pub struct DoloopState {
    minval: i32,
    maxval: i32,
    firsttime: bool,
    depth: i32,
}

impl DoloopState {
    pub fn new(min: i32, max: i32) -> Self {
        DoloopState {
            minval: min,
            maxval: max,
            firsttime: true,
            depth: 1,
        }
    }

    pub fn doloop(&mut self, indices: &mut [i32]) -> i32 {
        let nloops = indices.len();

        if self.firsttime {
            for i in 0..nloops {
                indices[i] = self.minval + i as i32;
            }
            self.firsttime = false;

            return if self.maxval >= self.minval + nloops as i32 {
                self.minval + nloops as i32 - 1
            } else {
                self.minval - 1
            };
        }

        let current_level = nloops - 1;
        indices[current_level] += 1;

        if indices[current_level] <= (self.maxval - self.depth) {
            indices[current_level]
        } else if current_level == 0 {
            self.minval - self.depth
        } else {
            self.depth += 1;
            let result = self.doloop(&mut indices[0..current_level]);
            indices[current_level] = result + 1;
            self.depth -= 1;
            indices[current_level]
        }
    }
}
