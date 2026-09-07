#ifndef SOFTMAX_H
#define SOFTMAX_H

namespace mytorch {
    class Attention {
        public:
            Attention(int model_dim, int num_head);
        private:
            int d_model;
            int n_head;
    };
}
#endif 