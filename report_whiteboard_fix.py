# Lines 500-540 replacement for report_whiteboard.py
    add(_band(t1_block, t2_block))
    add(_sp())

    # Band row 3: Fig 4 (left) | Table 3 (right)
    fig4_block = [_img(FIG4, COL_W, FIG_H),
                  _cap("Figure 4. Block volume overlay distribution.")]
    t3_block = ([_cap("Table 3. DFN calibration summary."),
                 _center_tbl(_df_table(cd, cw3, site_col="Site"))]
                if not cd.empty else [Paragraph("<i>No data.</i>", sNA)])
    add(_band(fig4_block, t3_block))
    add(_sp())
    
    # Band row 4: P32 calibration plots (2x2 grid) if available
    calib_plot_dir = os.path.join(SCRIPT_DIR, "outputs", "VARENNE", "02_calibration", "plots")
    if os.path.isdir(calib_plot_dir):
        calib_plots = sorted([
            os.path.join(calib_plot_dir, f)
            for f in os.listdir(calib_plot_dir)
            if f.startswith("P32_vs_P21_fam") and f.endswith(".png")
        ])
        if calib_plots:
            add(Paragraph("2.1 P32-P21 Calibration Curves", sSubsec))
            add(_sp())
            
            PLOT_W = COL_W * 0.95
            PLOT_H = PLOT_W * 0.68
            
            # Add plots in pairs (2-column layout)
            for i in range(0, len(calib_plots), 2):
                left_idx = i
                right_idx = i + 1
                
                left_plot = calib_plots[left_idx] if left_idx < len(calib_plots) else None
                right_plot = calib_plots[right_idx] if right_idx < len(calib_plots) else None
                
                left_content = ([_img(left_plot, PLOT_W, PLOT_H),
                                _cap(f"Figure 5.{left_idx+1}. P32 vs P21 calibration fam{left_idx+1}.")] 
                               if left_plot else [Spacer(1, 1)])
                right_content = ([_img(right_plot, PLOT_W, PLOT_H),
                                 _cap(f"Figure 5.{right_idx+1}. P32 vs P21 calibration fam{right_idx+1}.")] 
                                if right_plot else [Spacer(1, 1)])
                
                add(_band(left_content, right_content))
                add(_sp())

    # APPENDIX section continues...
