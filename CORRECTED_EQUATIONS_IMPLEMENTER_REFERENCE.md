# Corrected equations: implementation reference for a second-order Kerr Teukolsky solver

**Specification version:** 2.0.0  
**Date:** 2026-08-11  
**Legacy snapshot audited:** `JLRipley314/teuk-fortran-2020`, commit `0a12faef3461f04ef71c8d42517eaaba512e34aa`

This document is the compact implementation reference. The longer derivation document explains how each equation was obtained and independently checked. The executable audit is `verify_second_order_teukolsky.py`; the same equations are also encoded in `equation_spec_v2.yaml` and `SOURCE_TERM_LEDGER.csv`.

## 1. Conventions that must be fixed before coding

Use the horizon-penetrating, hyperboloidally compactified coordinates of Ripley et al.:

\[
r=\frac{L^2}{R},\qquad R=0\ \text{at future null infinity},\qquad R_H=\frac{L^2}{r_+}.
\]

Decompose every field as

\[
X(T,R,\theta,\phi)=\sum_m X_m(T,R,\theta)e^{im\phi}.
\]

For mode conjugation define

\[
X_m^{\sharp}\equiv \overline{X_{-m}}.
\]

Do not replace this rule with `conj(X_m)` unless a real-field relation has been proved for the specific NP/tetrad component. Store all signed modes explicitly and evaluate every ordered source pair

\[
(m_1,m_2)\longrightarrow m_t=m_1+m_2.
\]

Internally set \(M=1\) unless the entire code uses a dimensionally consistent strong-unit system. Convert only at input/output.

## 2. Spin, boost, and radial rescalings

Use the following rescaled first-order fields:

\[
\begin{array}{c|c|c|c|c}
\text{physical field}&\text{rescaled field}&s&b&\text{definition}\\ \hline
\Psi_4^{(1)}&F&-2&-2&\Psi_4^{(1)}=RF\\
\Psi_3^{(1)}&G&-1&-1&\Psi_3^{(1)}=R^2G\\
\Psi_2^{(1)}&H&0&0&\Psi_2^{(1)}=R^3H\\
\lambda^{(1)}&\Lambda&-2&-1&\lambda^{(1)}=R\Lambda\\
\pi^{(1)}&\Pi&-1&0&\pi^{(1)}=R^2\Pi\\
h_{\bar m\bar m}&B&-2&0&h_{\bar m\bar m}=RB\\
h_{l\bar m}&C&-1&1&h_{l\bar m}=R^2C\\
\mu h_{ll}&U&0&1&\mu h_{ll}=R^3U
\end{array}
\]

The last line is deliberate: the evolved variable is \(\mu h_{ll}\), not \(h_{ll}\) itself. Since \(\mu=R\mu_0\), this implies

\[
h_{ll}=\frac{R^2U}{\mu_0}.
\]

The needed background quantities are

\[
\mu=R\mu_0,
\qquad
\mu_0=\frac{1}{-L^2+iaR\cos\theta},
\]

\[
\tau=R^2\tau_0,
\qquad
\tau_0=\frac{ia\sin\theta}{\sqrt2\,(L^2-iaR\cos\theta)^2},
\]

\[
\pi=R^2\pi_0,
\qquad
\pi_0=-\frac{ia\sin\theta}{\sqrt2\,[L^4+a^2R^2\cos^2\theta]},
\]

\[
\Psi_2^{(0)}=R^3\psi_{20},
\qquad
\psi_{20}=-\frac{M}{(L^2-iaR\cos\theta)^3}.
\]

A useful exact identity is

\[
\boxed{\Delta\mu=-\mu^2.}
\]

It gives

\[
(\Delta-\mu+\bar\mu)h_{ll}
=\frac{1}{\mu}\Delta(\mu h_{ll})+\bar\mu h_{ll}.
\]

## 3. Rescaled GHP operators

For a field \(X\) with radial falloff \(R^n\), define

\[
\Delta_n X\equiv R^{-n}\Delta(R^nX),
\]

\[
\eth_nX\equiv R^{-(n+1)}\eth(R^nX),
\qquad
\eth'_nX\equiv R^{-(n+1)}\eth'(R^nX).
\]

In the coordinates used here,

\[
\boxed{
\Delta_nX=
\left(2+\frac{4MR}{L^2}\right)\partial_TX
+\frac{R}{L^2}\left(R\partial_RX+nX\right).
}
\]

Let \(s\) and \(b\) be spin and boost weights, and define \(p=s+b\), \(q=-s+b\). If \(\mathcal R_s\) and \(\mathcal L_s\) are the unit-sphere spin-raising and lowering matrices,

\[
\mathcal R_s\,{}_sY_{\ell m}
=\sqrt{(\ell-s)(\ell+s+1)}\,{}_{s+1}Y_{\ell m},
\]

\[
\mathcal L_s\,{}_sY_{\ell m}
=-\sqrt{(\ell+s)(\ell-s+1)}\,{}_{s-1}Y_{\ell m},
\]

then

\[
\boxed{
\eth_nX=
\frac{-ia\sin\theta\,\partial_TX+\mathcal R_sX}
{\sqrt2\,(L^2-iaR\cos\theta)}
-\frac{ipaR\sin\theta}
{\sqrt2\,(L^2-iaR\cos\theta)^2}X,
}
\]

\[
\boxed{
\eth'_nX=
\frac{ia\sin\theta\,\partial_TX+\mathcal L_sX}
{\sqrt2\,(L^2+iaR\cos\theta)}
+\frac{iqaR\sin\theta}
{\sqrt2\,(L^2+iaR\cos\theta)^2}X.
}
\]

The label \(n\) is radial-falloff bookkeeping; the displayed angular formulas have no explicit \(n\) because the angular tetrad vector contains no radial derivative.

## 4. Correct continuum second-order source

The second-order field obeys

\[
\boxed{\mathcal T\Psi_4^{(2)}=\mathcal S,}
\]

with the same background spin-\(-2\) Teukolsky operator as at first order and

\[
\boxed{
\mathcal S=(\Delta+4\mu+\bar\mu)s_d
+(\eth'+4\pi-\bar\tau)s_t.
}
\]

The correct inner sources in outgoing radiation gauge are

\[
\begin{aligned}
s_d={}&
\frac12h_{ll}(\Delta+\mu)\Psi_4^{(1)}\\
&+\Psi_4^{(1)}\left[
\frac12(\eth+\bar\pi+2\tau)h_{l\bar m}
+(\Delta-\mu+\bar\mu)h_{ll}
-\frac12(\eth'-5\pi-4\bar\tau)h_{lm}
\right]\\
&-\frac12\Psi_3^{(1)}\left[
(\eth+\bar\pi+\tau)h_{\bar m\bar m}
+(\Delta-2\mu+\bar\mu)h_{l\bar m}
\right]\\
&-\left(h_{l\bar m}\Delta
-\frac12h_{\bar m\bar m}\eth
-4\pi^{(1)}\right)\Psi_3^{(1)}
-3\lambda^{(1)}\Psi_2^{(1)},
\end{aligned}
\]

\[
\begin{aligned}
s_t={}&
-h_{lm}(\Delta+\mu+2\bar\mu)\Psi_4^{(1)}
+\frac12h_{mm}\eth'\Psi_4^{(1)}\\
&+\Psi_4^{(1)}\left[
\bar\pi^{(1)}-\Delta h_{lm}
+\left(\eth'-\frac12\pi-\frac12\bar\tau\right)h_{mm}
\right].
\end{aligned}
\]

### Critical correction to the legacy code

The factor \(1/2\) multiplies the **entire** operator combination

\[
\boxed{
\frac12(\eth+\bar\pi+2\tau)h_{l\bar m}.
}
\]

After radial rescaling, the correct two pieces are

\[
\boxed{
\frac12R\,\eth_2C
+\frac12R^2(\bar\pi_0+2\tau_0)C.
}
\]

The legacy source uses coefficient \(1/2\) on the first piece and coefficient \(1\) on the second. Its excess is

\[
\boxed{
\delta D_{\rm legacy}
=\frac12R^2F_1(\bar\pi_0+2\tau_0)C_2.
}
\]

This is nonzero for rotating Kerr. At \(\cos\theta=0\),

\[
\bar\pi_0+2\tau_0=\frac{3ia}{\sqrt2L^4}.
\]

## 5. Correct compactified ordered-pair source

Define

\[
D\equiv \frac{s_d}{R^3},\qquad T\equiv\frac{s_t}{R^3}.
\]

For one ordered pair \((m_1,m_2)\), with target \(m_t=m_1+m_2\), the corrected expression is

\[
\begin{aligned}
D_{12}={}&
\frac12U_1\left(\frac{\Delta_1F_2}{\mu_0}+RF_2\right)\\
&+F_1\Bigg[
\frac12R\,\eth_2C_2
+\frac12R^2(\bar\pi_0+2\tau_0)C_2
+\frac{\Delta_3U_2}{\mu_0}
+R\,U_2^{\sharp}\\
&\hspace{3.8em}
-\frac12R\,\eth'_2C_2^{\sharp}
+\frac12R^2(5\pi_0+4\bar\tau_0)C_2^{\sharp}
\Bigg]\\
&-\frac12G_1\Bigg[
R\,\eth_1B_2
+R^2(\bar\pi_0+\tau_0)B_2
+R\,\Delta_2C_2
+R^2(-2\mu_0+\bar\mu_0)C_2
\Bigg]\\
&-R\,C_1\Delta_2G_2
+\frac12R\,B_1\eth_2G_2
+4R\,\Pi_1G_2
-3R\,\Lambda_1H_2.
\end{aligned}
\]

The transverse inner source is

\[
\begin{aligned}
T_{12}={}&
-C_1^{\sharp}\left[
\Delta_1F_2+R(\mu_0+2\bar\mu_0)F_2
\right]
+\frac12B_1^{\sharp}\eth'_1F_2\\
&+F_1\left[
\Pi_2^{\sharp}
-\Delta_2C_2^{\sharp}
+\eth'_1B_2^{\sharp}
-\frac12R(\pi_0+\bar\tau_0)B_2^{\sharp}
\right].
\end{aligned}
\]

Sum all ordered pairs:

\[
D_{m_t}=\sum_{m_1+m_2=m_t}D_{12},
\qquad
T_{m_t}=\sum_{m_1+m_2=m_t}T_{12}.
\]

Then

\[
\boxed{
\frac{\mathcal S_{m_t}}{R^3}
=\Delta_3D_{m_t}
+R(4\mu_0+\bar\mu_0)D_{m_t}
+R\eth'_3T_{m_t}
+R^2(4\pi_0-\bar\tau_0)T_{m_t}.
}
\]

The forcing appearing in the coordinate \((P,Q,\psi)\) equation is

\[
\boxed{
\mathcal F^{(2)}_{m_t}
=2\left(L^4+a^2R^2\cos^2\theta\right)
\frac{\mathcal S_{m_t}}{R^3}.
}
\]

## 6. First-order metric reconstruction

The physical outgoing-radiation-gauge transport equations are

\[
-(\Delta+4\mu)\Psi_3^{(1)}+(\eth-\tau)\Psi_4^{(1)}=0,
\]

\[
-(\Delta+\mu+\bar\mu)\lambda^{(1)}-\Psi_4^{(1)}=0,
\]

\[
-(\Delta+3\mu)\Psi_2^{(1)}+(\eth-2\tau)\Psi_3^{(1)}=0,
\]

\[
-(\Delta-\mu+\bar\mu)h_{\bar m\bar m}-2\lambda^{(1)}=0,
\]

\[
-\Delta\pi^{(1)}-\Psi_3^{(1)}
-(\bar\pi+\tau)\lambda^{(1)}
+\frac12\mu(\bar\pi+\tau)h_{\bar m\bar m}=0,
\]

\[
-(\Delta+\bar\mu)h_{l\bar m}
-2\pi^{(1)}-\tau h_{\bar m\bar m}=0,
\]

\[
\begin{aligned}
0={}&-(\Delta+\bar\mu)(\mu h_{ll})
-\mu(\eth+\bar\pi+2\tau)h_{l\bar m}
-2\Psi_2^{(1)}\\
&-\pi(\eth'-\pi)h_{mm}
+(\mu\eth'-3\mu\pi+2\bar\mu\pi-2\mu\bar\tau)h_{lm}\\
&-2(\eth+\bar\pi)\pi^{(1)}-2\pi\bar\pi^{(1)}.
\end{aligned}
\]

Their rescaled form is

\[
\boxed{\Delta_2G=-4R\mu_0G+\eth_1F-R\tau_0F,}
\]

\[
\boxed{\Delta_1\Lambda=-R(\mu_0+\bar\mu_0)\Lambda-F,}
\]

\[
\boxed{\Delta_3H=-3R\mu_0H+\eth_2G-2R\tau_0G,}
\]

\[
\boxed{\Delta_1B=R(\mu_0-\bar\mu_0)B-2\Lambda,}
\]

\[
\boxed{
\Delta_2\Pi=-G-R(\bar\pi_0+\tau_0)\Lambda
+\frac12R^2\mu_0(\bar\pi_0+\tau_0)B,
}
\]

\[
\boxed{\Delta_2C=-R\bar\mu_0C-2\Pi-R\tau_0B,}
\]

\[
\boxed{
\begin{aligned}
\Delta_3U={}&-R\bar\mu_0U
-R\mu_0\eth_2C
-R^2\mu_0(\bar\pi_0+2\tau_0)C
-2\eth_2\Pi
-2R\bar\pi_0\Pi
-2H\\
&-2R\pi_0\Pi^{\sharp}
-R\pi_0\eth'_1B^{\sharp}
+R^2\pi_0^2B^{\sharp}
+R\mu_0\eth'_2C^{\sharp}\\
&+R^2(-3\mu_0\pi_0+2\bar\mu_0\pi_0-2\mu_0\bar\tau_0)C^{\sharp}.
\end{aligned}
}
\]

For each equation solve \(\Delta_nX=R_X\) as

\[
\partial_TX=
\frac{R_X-(R^2/L^2)\partial_RX-(nR/L^2)X}
{2+4MR/L^2}.
\]

## 7. Linear and driven Teukolsky first-order reduction

For spin weight \(s\) and Fourier mode \(m\), define

\[
C_T=8M\left(2M-\frac{a^2R}{L^2}\right)
\left(1+\frac{2MR}{L^2}\right)-a^2\sin^2\theta,
\]

\[
K=L^2-\frac{(8M^2-a^2)R^2}{L^2}
+\frac{4a^2MR^3}{L^4},
\]

\[
H_R=\frac{R^2}{L^4}(L^4-2L^2MR+a^2R^2),
\]

\[
\begin{aligned}
G_m={}&2iam\left(1+\frac{4MR}{L^2}\right)\\
&+2\left[
2M\left(-s+(2+s)\frac{2MR}{L^2}-\frac{3a^2R^2}{L^4}\right)
-\frac{a^2R}{L^2}+isa\cos\theta
\right].
\end{aligned}
\]

Define

\[
P=C_T\partial_T\psi-2K\partial_R\psi+G_m\psi,
\qquad Q=\partial_R\psi.
\]

Then

\[
\boxed{
\partial_T\psi=\frac{P+2KQ-G_m\psi}{C_T}.
}
\]

The \(P\) equation is

\[
\begin{aligned}
\partial_TP={}&H_R\partial_RQ\\
&+\left[
2R\left(1+s-(3+s)\frac{MR}{L^2}+\frac{2a^2R^2}{L^4}\right)
-\frac{2iamR^2}{L^2}
\right]Q\\
&+\left[
-2R\left((1+s)\frac{M}{L^2}-\frac{a^2R}{L^4}\right)
-\frac{2iamR}{L^2}
\right]\psi
+{}_{s}\!\Delta_{\Omega}\,\psi+\mathcal F.
\end{aligned}
\]

Use \(\mathcal F=0\) at first order and the corrected \(\mathcal F^{(2)}\) above at second order.

Do not copy the expanded legacy \(Q\) coefficients. Generate

\[
\boxed{
\partial_TQ=
\partial_R\left(\frac{P+2KQ-G_m\psi}{C_T}\right)
-\gamma_Q(Q-\partial_R\psi),
}
\]

or impose \(Q=\partial_R\psi\) at every RK stage. Implement both strategies and cross-check them.

## 8. Nonlinear angular product oracle

For a product with \(s_3=s_1+s_2\) and \(m_3=m_1+m_2\), use the generalized spin-weighted Gaunt coefficient

\[
\begin{aligned}
&\int {}_{s_1}Y_{\ell_1m_1}
{}_{s_2}Y_{\ell_2m_2}
\overline{{}_{s_3}Y_{\ell_3m_3}}\,d\Omega\\
&=(-1)^{m_3+s_3}
\sqrt{\frac{(2\ell_1+1)(2\ell_2+1)(2\ell_3+1)}{4\pi}}
\begin{pmatrix}\ell_1&\ell_2&\ell_3\\-s_1&-s_2&s_3\end{pmatrix}
\begin{pmatrix}\ell_1&\ell_2&\ell_3\\m_1&m_2&-m_3\end{pmatrix}.
\end{aligned}
\]

The production padded-collocation product may be faster, but it must agree with this modal oracle over the supported bandlimit. Overcollocation without this test is not sufficient evidence of dealiasing.

## 9. Required fourth-order stage coupling

The legacy driven scheme uses

\[
S_1=S_n,\qquad
S_2=S_3=\frac{S_n+S_{n+1}}2,\qquad
S_4=S_{n+1}.
\]

Its RK4 source contribution is exactly

\[
\frac{\Delta T}{6}(S_1+2S_2+2S_3+S_4)
=\frac{\Delta T}{2}(S_n+S_{n+1}),
\]

so the driven solution is only globally second order for a time-dependent source.

The new solver must evaluate, at every common RK stage:

1. the first-order Teukolsky RHS;
2. the seven reconstruction RHSs;
3. all inner-source primitives \(D\) and \(T\);
4. the time derivatives needed inside \(\Delta\), \(\eth\), and \(\eth'\);
5. the outer source \(\mathcal S\);
6. the second-order Teukolsky RHS.

Because the background system is stationary and linear, at a stage state \(U\),

\[
\dot U=L(U),\qquad \ddot U=L(\dot U).
\]

For every bilinear source primitive \(B(U,V)\), evaluate its tangent by

\[
\boxed{
\partial_TB(U,V)=B(\dot U,V)+B(U,\dot V).
}
\]

A `Jet1<T>{value, dt}` representation or a manually generated Jacobian-vector product is appropriate. Start with classical coupled RK4 as the correctness oracle. Optimize to a low-storage scheme only after fourth-order convergence is demonstrated for the full driven system.

## 10. Numerical method requirements for the instability study

Use finite differences only in \(R\), retaining the spin-weighted spectral angular representation and explicit \(m\) decomposition.

Recommended progression:

- a verified diagonal-norm SBP operator with SAT boundary treatment as the reference discretization;
- fourth-order interior/second-order boundary closure first, followed by sixth-order interior/third-order boundary closure;
- compatible KO/SBP dissipation included in the method-of-lines RHS;
- no post-step filter followed by reuse of a pre-filter RHS;
- optional two-block or mapped near-horizon refinement before nonlinear WENO/MP5;
- nonlinear radial stencils only after convergence evidence shows an actual loss of smoothness rather than merely large transverse derivatives.

For exact extremality and near extremality, separately output

\[
\Psi_4^{(1)},\quad \Psi_4^{(2)},\quad
\partial_R^n\Psi_4^{(1,2)}\quad (n\ge 4),
\]

all source families and angular modes, horizon-corotating/demodulated fields, reconstruction and reduction residuals, and the near-extremal scaling variable \(\kappa v\).

Raw horizon \(\Psi_4^{(2)}\) in the code tetrad is a convention-fixed diagnostic, not generally a first-order gauge/tetrad invariant. Include a regular horizon tetrad or infalling-observer diagnostic before making a physical instability claim.

## 11. Minimal regression tests that must exist before production runs

1. The executable symbolic audit passes.
2. A focused test fails when the corrected connection coefficient \(1/2\) is changed to \(1\).
3. A forced manufactured problem shows fourth-order time convergence; the legacy staging reproduces order two.
4. The source scales as amplitude squared.
5. Every source term has the correct spin/boost weight and radial falloff.
6. Ordered \(m\)-pair enumeration is deterministic and includes both orderings.
7. Padded angular products agree with the Gaunt oracle.
8. The Schwarzschild limit removes all disputed Kerr connection terms.
9. Constraint and reconstruction residuals converge at the designed order.
10. CPU reference and Kokkos CPU/GPU kernels agree within a justified tolerance.
11. Restarted and uninterrupted evolutions agree.
12. Exact-extremal and near-extremal test problems are run separately.
