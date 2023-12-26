    .section .note.GNU-stack,""
    .global matmul_d
    .global matmul_f
    .text


# rdi - m ; rsi - n ; rdx: - p; rcx: - a; r8 - b; r9 - c
#function
matmul_d:
    push %r12
    push %r13
    push %r14
    push %r15

    mov $0, %r10            # 0 .. m
matmul_d_mloop:
    cmp %rdi, %r10
    jae matmul_d_mloop_ret

    mov $0, %r11            # 0 .. p
matmul_d_ploop:
    cmp %rdx, %r11
    jae matmul_d_ploop_ret

    mov $0, %r14            # 0 .. n
    pxor %xmm2, %xmm2    
    mov %rsi, %r12
    imul %r10, %r12                 # idx1=n*row
    mov %rsi, %r13
    imul %r11, %r13                 # idx2=n*col    
matmul_d_nloop:
    cmp %rsi, %r14
    jae matmul_d_nloop_ret

    vmovupd (%rcx, %r12, 8), %ymm0
    vmulpd (%r8, %r13, 8), %ymm0, %ymm0

    vextractf128 $0x1, %ymm0, %xmm1
    vaddpd %xmm1, %xmm0, %xmm0
    vshufpd $0b01, %xmm0, %xmm0, %xmm1
    addsd %xmm1, %xmm0

    addsd %xmm0, %xmm2

    add $4, %r14
    add $4, %r12
    add $4, %r13
    jmp matmul_d_nloop

matmul_d_nloop_ret:

    mov %rdx, %r15
    imul %r10, %r15
    add %r11, %r15                  # r15 = p * row + col
    #vshufpd $0b01, %xmm2, %xmm2, %xmm2
    movq %xmm2, (%r9, %r15, 8)

    inc %r11
    jmp matmul_d_ploop
matmul_d_ploop_ret:

    inc %r10                
    jmp matmul_d_mloop
matmul_d_mloop_ret:
    
    pop %r15
    pop %r14
    pop %r13
    pop %r12

    rep; ret

# rdi - m ; rsi - n ; rdx: - p; rcx: - a; r8 - b; r9 - c
#function
matmul_f:
    push %r12
    push %r13
    push %r14
    push %r15

    mov $0, %r10            # 0 .. m
matmul_f_mloop:
    cmp %rdi, %r10
    jae matmul_f_mloop_ret

    mov $0, %r11            # 0 .. p
matmul_f_ploop:
    cmp %rdx, %r11
    jae matmul_f_ploop_ret

    mov $0, %r14            # 0 .. n
    pxor %xmm2, %xmm2    
    mov %rsi, %r12
    imul %r10, %r12                 # idx1=n*row
    mov %rsi, %r13
    imul %r11, %r13                 # idx2=n*col    
matmul_f_nloop:
    cmp %rsi, %r14
    jae matmul_f_nloop_ret

    vmovups (%rcx, %r12, 4), %ymm0
    vmulps (%r8, %r13, 4), %ymm0, %ymm0

    vextractf128 $0x1, %ymm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0b00001110, %xmm0, %xmm0, %xmm1
    vaddps %xmm1, %xmm0, %xmm0
    vshufps $0b00000001, %xmm0, %xmm0, %xmm1
    vaddss %xmm1, %xmm0, %xmm0    

    addss %xmm0, %xmm2

    add $8, %r14
    add $8, %r12
    add $8, %r13
    jmp matmul_f_nloop

matmul_f_nloop_ret:

    mov %rdx, %r15
    imul %r10, %r15
    add %r11, %r15                  # r15 = p * row + col
    #vshufpd $0b01, %xmm2, %xmm2, %xmm2
    movd %xmm2, (%r9, %r15, 4)

    inc %r11
    jmp matmul_f_ploop
matmul_f_ploop_ret:

    inc %r10                
    jmp matmul_f_mloop
matmul_f_mloop_ret:
    
    pop %r15
    pop %r14
    pop %r13
    pop %r12

    rep; ret

