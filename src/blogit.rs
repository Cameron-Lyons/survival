// Bounded link functions

struct LinkFunctionParams {
    edge: f64,
}

impl LinkFunctionParams {
    fn new(edge: f64) -> Self {
        LinkFunctionParams { edge }
    }
    fn blogit(&self, input: f64) -> f64 {
        let adjusted_input = if input < self.edge {
            self.edge
        } else if input > 1.0 - self.edge {
            1.0 - self.edge
        } else {
            input
        };
        adjusted_input.ln() - (1.0 - adjusted_input).ln()
    }

    fn bprobit(&self, input: f64) -> f64 {
        let adjusted_input = if input < self.edge {
            self.edge
        } else if input > 1.0 - self.edge {
            1.0 - self.edge
        } else {
            input
        };
        adjusted_input.probit() - (1.0 - adjusted_input).probit()
    }

    fn bcloglog(&self, input: f64) -> f64 {
        let adjusted_input = if input < self.edge {
            self.edge
        } else if input > 1.0 - self.edge {
            1.0 - self.edge
        } else {
            input
        };
        adjusted_input.cloglog() - (1.0 - adjusted_input).cloglog()
    }

    fn blog(&self, input: f64) -> f64 {
        let adjusted_input = if input < self.edge {
            self.edge
        } else {
            input
        };
    
}
