! This subroutine extracts skewers from a 3D density field and computes the Lyman-alpha flux using the
! Fluctuating Gunn-Peterson Approximation (FGPA). The flux is computed as exp(-tau), where tau is the optical depth
! given by tau = A * (1 + delta)^beta, with A being the amplitude and beta being the power-law index.
subroutine fgpa_skewer_transform(res, n_skewers, n_los, skewer_x, skewer_y, delta_field, flux, log_a, beta, rho_floor)
    implicit none
    integer, intent(in) :: res
    integer, intent(in) :: n_skewers
    integer, intent(in) :: n_los
    integer, intent(in) :: skewer_x(n_skewers)
    integer, intent(in) :: skewer_y(n_skewers)
    real(4), intent(in) :: delta_field(res * res * res)
    real(4), intent(out) :: flux(n_skewers * n_los)
    real(4), intent(in) :: log_a
    real(4), intent(in) :: beta
    real(4), intent(in) :: rho_floor

    integer :: s
    integer :: z
    integer :: ix
    integer :: iy
    integer :: field_idx
    integer :: flux_idx
    real(4) :: amp
    real(4) :: rho
    real(4) :: tau

    amp = exp(log_a)
    do s = 1, n_skewers
        ix = skewer_x(s)
        iy = skewer_y(s)
        do z = 1, n_los
            field_idx = (z - 1) + iy * res + ix * res * res + 1
            flux_idx = z + (s - 1) * n_los
            rho = 1.0e0 + delta_field(field_idx)
            if (rho < rho_floor) then
                rho = rho_floor
            end if
            tau = amp * rho ** beta
            flux(flux_idx) = exp(-tau)
        end do
    end do
end subroutine fgpa_skewer_transform
